import logging
import numpy as np
import sys
import torch
from PIL import Image
from tqdm import tqdm
from vllm.multimodal.utils import fetch_image
import os
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from typing import Union

from retrievals.base_retriever import BaseRetriever

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)

class UnimodalRetriever(BaseRetriever):
    def __init__(
            self,
            text_model_name:str ="BAAI/bge-large-en-v1.5",
            image_model_name:str="openai/clip-vit-large-patch14-336",
            device: str = 'cuda'
        ):
        super().__init__()
        # Load model and processor

        self.web_search = True if text_model_name is not None else False
        if self.web_search:
            self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)    

        self.image_model = CLIPModel.from_pretrained(image_model_name).to(device)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.device = device

    def extract_features_from_textEncoder(self,text: Union[str, list[str]]) -> np.ndarray:
        """
        Extract features from text using the model and tokenizer.

        Args:
            text: The input text

        Returns:
            numpy.ndarray: The extracted features
        """
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        # Tokenize and prepare inputs
        inputs = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use mean pooling
            attention_mask = inputs["attention_mask"]
            features = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()

        if single_input:
            return features[0]
        return features

    def extract_features_from_clip(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from image using the CLIP model and processor.

        Args:
            image: The input image

        Returns:
            numpy.ndarray: The extracted features
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.image_model.get_image_features(**inputs)

        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def mean_pooling(
        self,token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings using attention mask.

        Args:
            token_embeddings: Token embeddings from model
            attention_mask: Attention mask from tokenizer

        Returns:
            torch.Tensor: Mean pooled embeddings
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_input(self, inputs:dict[str]):
        """
        Encode a batch of images (and/or texts) into embeddings using the MLLM.
        Args:
            inputs: list of PIL.Image.Image or (image, text) tuples
        Returns:
            np.ndarray of embeddings, shape (N, D)
        """
        embeddings = []
        batch_images= []
        batch_texts = []    
        for text,image in zip(inputs.get('text'), inputs.get('image')):
            if image:
                if isinstance(image, str):
                    if image.startswith(('http', 'https', 'oss')):
                        try:
                            image_obj = fetch_image(image)
                        except Exception as e:
                            print(f"Warning: Failed to fetch image {image}: {e}")
                    else:
                        abs_image_path = os.path.abspath(image)
                        if os.path.exists(abs_image_path):
                            image_obj = Image.open(abs_image_path)
                        else:
                            print(f"Warning: Image file not found: {abs_image_path}")
                else:
                    image_obj = image
                batch_images.append(image_obj)
            else:
                batch_images.append(None)
            batch_texts.append(text)
            
        processor_inputs = {
            "text": batch_texts,
            "images": batch_images,
        }
        # Assume all batch inputs will be either text only or image only
        # filter out the None input for the corresponding modality
        if processor_inputs['text'] and all(text is not None for text in processor_inputs['text']) and self.web_search:
            embeddings = self.extract_features_from_textEncoder(
                processor_inputs['text'],
            )
        else: # if not web_search feature is on, text input will be calculated by clip-vit
            embeddings = self.extract_features_from_clip(
                processor_inputs['images'],
            )
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def search(self, query_images, candidate_embeddings, candidate_infos=None, topk=10):
        """
        Given a batch of query images, encode and search against candidate embeddings.
        Args:
            query_images: list of PIL.Image.Image
            candidate_embeddings: np.ndarray, shape (N, D)
            candidate_infos: list of dicts, optional, metadata for each candidate
            topk: int, number of results to return
        Returns:
            List of topk results for each query image
        """
        # Encode query images
        query_embs = self.encode_input(query_images)
        # Normalize embeddings
        query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
        candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        # Compute similarity
        sims = np.matmul(query_embs, candidate_embeddings.T)  # (num_queries, num_candidates)
        results = []
        for i in range(len(query_images)):
            top_indices = np.argsort(sims[i])[::-1][:topk]
            top_scores = sims[i][top_indices]
            if candidate_infos is not None:
                top_infos = [candidate_infos[idx] for idx in top_indices]
            else:
                top_infos = top_indices.tolist()
            results.append([
                {"score": float(score), "info": info}
                for score, info in zip(top_scores, top_infos)
            ])
        return results