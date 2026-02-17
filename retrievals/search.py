import os
from enum import Enum
from typing import Any, Union

import numpy as np
import requests
import torch
from PIL import Image
from transformers import Pipeline
from typing import Optional

from utils.utils import maybe_list, lookup_web_content
from .base_index import MockWeb, ImageKG
from retrievals.unimodal_retriever import UnimodalRetriever
from retrievals.qwen3_vl_retriever import Qwen3VLRetriever


class InputType(Enum):
    TEXT_QUERY = "text_query"
    IMAGE_PATH = "image_path"
    IMAGE_URL = "image_url"
    IMAGE_EMBEDDING = "image_embedding"
    IMAGE_OBJECT = "image_object"

class CustomSearchPipeline(Pipeline):
    def __init__(
        self,
        text_index_path: Optional[str] = None,
        image_index_path: Optional[str] = None,
        retriever_type : Optional[str] = None,
        text_model_name: Optional[str] = None,
        image_model_name: Optional[str] = None,
        multimodal_model_name: Optional[str] = None,
        web_hf_dataset_id  : Optional[str] = None,
        image_hf_dataset_id  : Optional[str] = None,
        web_hf_dataset_tag: Optional[str] = None,
        image_hf_dataset_tag: Optional[str] = None,
        **kwargs,
    ):
        # Determine the device to use
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        device = torch.device(device_str)
        print(f"Using device: {device}")
        self.retriever_type= retriever_type
        self.multimodal_model_name = multimodal_model_name

        model=None
        if retriever_type == "separate":
            model = UnimodalRetriever(
                text_model_name=text_model_name, #"BAAI/bge-large-en-v1.5",
                image_model_name=image_model_name, #"openai/clip-vit-large-patch14-336",
                device=device
            )
        elif retriever_type == "mllm":
            if multimodal_model_name == "Qwen/Qwen3-VL-Embedding-2B":
                model = Qwen3VLRetriever(
                    model_name= multimodal_model_name,
                    checkpoint_path = 'models/Qwen3-VL-Embedding-2B',
                    dtype ="bfloat16",
                )
            elif multimodal_model_name == "Qwen/Qwen3-VL-Embedding-8B":
                model = Qwen3VLRetriever(
                    model_name= multimodal_model_name,
                    checkpoint_path = 'models/Qwen3-VL-Embedding-8B',
                    dtype ="bfloat16",
                )
                model.device = device  # Explicitly set device attribute
        
        self.is_private = False

        if text_index_path:
            self.web_search = MockWeb(text_index_path=text_index_path, hf_dataset_id=web_hf_dataset_id, web_hf_dataset_tag=web_hf_dataset_tag)
        self.image_kg = ImageKG(image_index_path=image_index_path, hf_dataset_id=image_hf_dataset_id, image_hf_dataset_tag=image_hf_dataset_tag)
        self.image_collection = self.image_kg.vector_db

        # Initialize with minimal arguments
        dummy_module = torch.nn.Module()
        dummy_module.device = device

        super().__init__(
            model=dummy_module,  # Pass one of the models to satisfy the base class,
            framework="pt",
            device=device,
            dtype=torch.bfloat16,  # Use bfloat16 for better performance on GPUs
            **kwargs,
        )
        self.device = device
        self.model = model
        self.web_content_private=False

    def _sanitize_parameters(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        # Handle retrieval source
        if "retrieval_source" in kwargs:
            if kwargs["retrieval_source"] not in ["text", "image"]:
                raise ValueError("retrieval_source must be 'text' or 'image'")
            forward_params["retrieval_source"] = kwargs["retrieval_source"]
        else:
            forward_params["retrieval_source"] = "image"  # Default to text search

        if "k" in kwargs:
            forward_params["k"] = kwargs["k"]

        return preprocess_params, forward_params, postprocess_params

    def _get_all_image_urls(self):
        return list(set([v['image_url'] for v in self.image_kg.id2_data]))

    def preprocess(self, inputs: Union[str, Image.Image, dict[str, Any]]) -> dict[str, Any]:
        processed_inputs={}
        if isinstance(inputs, str):
            processed_inputs[InputType.TEXT_QUERY] = inputs
        elif isinstance(inputs, Image.Image):
            if inputs.mode != "RGB":
                inputs = inputs.convert("RGB")
            processed_inputs[InputType.IMAGE_OBJECT] = inputs
        
        elif isinstance(inputs, dict):
            if "text" in inputs and inputs["text"]:
                processed_inputs[InputType.TEXT_QUERY] = inputs["text"]
            if "image" in inputs and inputs["image"]:
                if isinstance(inputs["image"], Image.Image):
                    processed_inputs[InputType.IMAGE_OBJECT] = inputs["image"].convert("RGB")
                elif os.path.isfile(inputs["image"]):
                    image = Image.open(inputs["image"]).convert("RGB")
                    processed_inputs[InputType.IMAGE_PATH] = image
                elif inputs["image"].startswith(("http://", "https://")):
                    try:
                        image = Image.open(requests.get(inputs["image"], stream=True).raw).convert("RGB")
                        processed_inputs[InputType.IMAGE_URL] = image
                    except Exception as e:
                        raise ValueError(f"Failed to load image from URL: {e}")
                
                    processed_inputs[InputType.IMAGE_OBJECT] = inputs["image"]
            
                elif isinstance(inputs["image"], np.ndarray):
                    processed_inputs[InputType.IMAGE_EMBEDDING] = inputs["image"]
        else:
            raise ValueError(
                f"Unsupported input type: {type(inputs)}. Provide text, image, or a dict with 'text' and/or 'image'."
            )
        
        return processed_inputs
    
    def _forward(self, input_dict: dict[InputType, Any], retrieval_source: str="image", k: int = 5):
        # Input will be given with batch_size=1
        processor_inputs = {'text': [], 'image': []}

        # Determine input types
        has_text = InputType.TEXT_QUERY in input_dict
        has_image = any(t in input_dict for t in [InputType.IMAGE_PATH, InputType.IMAGE_URL, InputType.IMAGE_OBJECT])
        has_embedding = InputType.IMAGE_EMBEDDING in input_dict

        # Prepare processor inputs
        text_input = input_dict.get(InputType.TEXT_QUERY)
        image_input = (
            input_dict.get(InputType.IMAGE_PATH) or
            input_dict.get(InputType.IMAGE_URL) or
            input_dict.get(InputType.IMAGE_OBJECT)
        )
    
        if has_embedding:
            # Use provided embedding directly
            qry_output = input_dict[InputType.IMAGE_EMBEDDING]
        else:
            # Prepare inputs for the model
            if has_text and has_image:
                if self.multimodal_model_name in ["Qwen/Qwen3-VL-Embedding-2B","Qwen/Qwen3-VL-Embedding-8B"]:
                    selected_instruction = "Represent the user's input."
                
                processor_inputs['text'].append(text_input)
                processor_inputs['image'].append(image_input)
            # Not related for our case
            elif has_text:
                if self.multimodal_model_name in ["Qwen/Qwen3-VL-Embedding-2B","Qwen/Qwen3-VL-Embedding-8B"]:
                    selected_instruction = "Represent the user's input."
                
                processor_inputs['text'].append(text_input)
                processor_inputs['image'].append(None)
            elif has_image:
                if self.multimodal_model_name in ["Qwen/Qwen3-VL-Embedding-2B","Qwen/Qwen3-VL-Embedding-8B"]:
                    selected_instruction = "Represent the user's input."

                processor_inputs['text'].append(text_input)
                processor_inputs['image'].append(image_input)              
            else:
                raise ValueError("No valid text or image input provided.")
            
        if self.retriever_type=="mllm":
            qry_output = self.model.encode_input(
                inputs=processor_inputs,
                instruction = selected_instruction,
            )
        elif self.retriever_type=="separate":
            # image retriever: "openai/clip-vit-large-patch14-336"
            # text encoder: "BAAI/bge-large-en-v1.5"
            qry_output = self.model.encode_input(
                inputs=processor_inputs,
            )
        else:
            raise NotImplementedError
        # Normalize query embedding
        if qry_output.ndim == 1:
            qry_output = qry_output.reshape(1, -1)
        
        # Perform retrieval based on source
        if retrieval_source == "image":
            if not self.image_collection:
                raise ValueError("Image collection not initialized")
            results = self.image_collection.query(
                query_embeddings=qry_output.tolist(),
                n_results=k
            )
            indices = [id for id in results["ids"][0]]
            distances = results["distances"][0] if "distances" in results else [1.0] * len(indices)
            scores = [1.0 - d for d in distances]
            retrieval_results = list(zip(indices, scores))
            return {"raw_results": retrieval_results, "type": "image"}

        elif retrieval_source == "text":
            if not self.web_search or not self.web_search.vector_db:
                raise ValueError("Text web index not initialized")
            results = self.web_search.vector_db.query(
                query_embeddings=qry_output.tolist(),
                n_results=k
            )
            if not results:
                raise ValueError("No results found")
            indices = [id for id in results["ids"][0]]
            # distances = results.get("distances", [1.0] * len(indices))
            distances = results["distances"][0] if "distances" in results else [1.0] * len(indices)
            scores = [1.0 - d for d in distances]

            # Deduplicate by page_url
            unique_results = {}
            for idx, score in zip(indices, scores):
                base_url = idx.split("_chunk")[0] if "_chunk" in idx else idx
                if base_url not in unique_results:
                    unique_results[base_url] = (idx, score)
            retrieval_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)[:k]
            return {"raw_results": retrieval_results, "type": "text"}

        else:
            raise ValueError(f"Invalid retrieval_source: {retrieval_source}")


    def postprocess(self, model_outputs: dict[str, Any]) -> list[dict[str, Any]]:
        results = model_outputs["raw_results"]
        if model_outputs["type"] == "text":
            if self.is_private:
                return [
                    {
                        **{
                            "index": ind,
                            "score": score,
                        },
                        **lookup_web_content(self.web_content_private, self.web_search.get_page_url(ind))
                    }
                    for ind, score in results
                ]
            else:
                return [
                    {
                        "index": ind,
                        "score": score,
                        "page_name": self.web_search.get_page_name(ind),
                        "page_snippet": self.web_search.get_page_snippet(ind),
                        "page_url": self.web_search.get_page_url(ind),
                    }
                    for ind, score in results
                ]
        elif model_outputs["type"] == "image":
            return [
                {
                    "index": ind,
                    "score": dist,
                    "url": self.image_kg.get_image_url(ind),
                    "entities": [
                        {
                            "entity_name": entity,
                            "entity_attributes": self.image_kg.get_entity(
                                entity_name=entity
                            ),
                        }
                        for entity in maybe_list(
                            self.image_kg.get_entity_name(ind)
                        )
                    ],
                }
                for ind, dist in results
            ]
        else:
            raise ValueError("Unknown model output type.")