from retrievals.base_retriever import BaseRetriever
import logging
import numpy as np
import os
from typing import List, Dict, Any
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
from PIL import Image
                            
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)

class Qwen3VLRetriever(BaseRetriever):
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen3-VL-Embedding-2B',
            checkpoint_path: str = 'models/Qwen3-VL-Embedding-2B', # should download and point to the local path of the model checkpoint
            dtype:str ="bfloat16",
            device: str = 'cuda'
        ):
        super().__init__()
        # Load model and processor
        engine_args = EngineArgs(
            model=checkpoint_path,
            runner="pooling",
            dtype=dtype,
            trust_remote_code=True,
            max_model_len = 166512 if model_name=='Qwen/Qwen3-VL-Embedding-8B' else None, # 
        )
        self.model = LLM(**vars(engine_args))
        self.device = device
        
    def format_input_to_conversation(self, input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> List[Dict]:
        content = []
        
        text = input_dict.get('text')
        image = input_dict.get('image')
        
        if image:
            image_content = None
            if isinstance(image, str):
                if image.startswith(('http', 'https', 'oss')):
                    image_content = image
                else:
                    abs_image_path = os.path.abspath(image)
                    image_content = 'file://' + abs_image_path
            else:
                image_content = image
            
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                })
        
        if text:
            content.append({'type': 'text', 'text': text})
        
        if not content:
            content.append({'type': 'text', 'text': ""})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        return conversation

    def encode_input(self, inputs: Dict[str, Any],instruction: str = "Represent the user's input.") -> Dict[str, Any]:
        """
        Encode a batch of images (and/or texts) into embeddings using the MLLM.
        Args:
            inputs: list of PIL.Image.Image or (image, text) tuples
            batch_size: batch size for encoding
        Returns:
            np.ndarray of embeddings, shape (N, D)
        """
        vllm_inputs = []
        for text,image in zip(inputs.get('text'), inputs.get('image')):
            input_dict = {
                "text": text,
                "image": image,
            }
            conversation = self.format_input_to_conversation(input_dict, instruction)
            
            prompt_text = self.model.llm_engine.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            multi_modal_data = None
            if image:
                if isinstance(image, str):
                    if image.startswith(('http', 'https', 'oss')):
                        try:
                            image_obj = fetch_image(image)
                            multi_modal_data = {"image": image_obj}
                        except Exception as e:
                            print(f"Warning: Failed to fetch image {image}: {e}")
                    else:
                        abs_image_path = os.path.abspath(image)
                        if os.path.exists(abs_image_path):
                            image_obj = Image.open(abs_image_path)
                            multi_modal_data = {"image": image_obj}
                        else:
                            print(f"Warning: Image file not found: {abs_image_path}")
                else:
                    multi_modal_data = {"image": image}
            
            result = {
                "prompt": prompt_text,
                "multi_modal_data": multi_modal_data
            }
            vllm_inputs.append(result)
            
        outputs = self.model.embed(vllm_inputs)
        embeddings_list = []
        for i, output in enumerate(outputs):
            emb = output.outputs.embedding
            embeddings_list.append(emb)
            print(f"Input {i} embedding shape: {len(emb)}")
        
        embeddings = np.array(embeddings_list)
        print(f"\nEmbeddings shape: {embeddings.shape}")
  
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