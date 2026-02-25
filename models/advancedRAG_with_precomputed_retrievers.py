from typing import Dict, List, Any
import json

from PIL import Image

import vllm
from openai import OpenAI
import io
import base64
from rich.console import Console
import logging
import torch

from models.base_agent import BaseAgent
from models.model import AgentResponse
from retrievals.crag_web_result_fetcher import WebSearchResult
from utils.utils import filter_entity_attributes

logger = logging.getLogger(__name__)
console = Console()

class PrecomputedRetrievalAdvancedRAG(BaseAgent):
    def __init__(
        self, 
        search_pipeline: None,
        model_name: str,
        rag_type : str,
    ):
        """
        Initialize the PrecomputedRetrievalAdvancedRAG.
        
        Args:
            search_pipeline: Pipeline for searching relevant information
            model_name: The model name to use for generation
            rag_type : str, specifying the type of RAG augmentation to use. Options include:
                - "image_only": Use only image retrieval results for augmentation.
                - "dino_cropped_image_only": Use cropped image regions from Grounding DINO for augmentation.
                - "gt_cropped_image_only": Use cropped image regions from GT retrieval for augmentation.
                - "text_only": Use only text retrieval results for augmentation.
                - "text_only_w_textual_only_query": Use text retrieval results with a textual-only version of the query for augmentation.
                - "both_normal_image_text": Use both image and text retrieval results for augmentation.
                - "both_dino_cropped_image_text": Use both cropped image regions from Grounding DINO and text retrieval results for augmentation.
                - "both_gt_cropped_image_text": Use both cropped image regions from GT retrieval and text retrieval results for augmentation.
                - "no_augmentation": Do not use any retrieval results for augmentation.
                - "no_augmentation_text_only": Do not use any retrieval results for augmentation and use a textual-only version of the query
        """
        super().__init__(search_pipeline)
        self.model_name = model_name
        # VLLM configuration
        self.max_gen_len = 64
        # GPU utilization settings 
        self.vllm_tensor_parallel_size=torch.cuda.device_count()
        self.vllm_gpu_memory_utilization=0.65
        # These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
        self.max_model_len = 8192
        self.max_num_seqs = 2
        self.max_generation_tokens=75
        ## Search configuration
        self.image_search_topk = 10
        self.text_search_topk = 10
        self.text_query_reformulation = True
        self.rag_type = rag_type
        
        ######## GT MODE
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the vLLM model, tokenizer, and Grounding DINO model with appropriate settings.
        """
        if 'gpt' in self.model_name:
            self.vllm = OpenAI()
        else:
            # Initialize the model with vLLM
            self.vllm = vllm.LLM(
                self.model_name,
                tensor_parallel_size=self.vllm_tensor_parallel_size, 
                gpu_memory_utilization=self.vllm_gpu_memory_utilization, 
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                trust_remote_code=True,
                dtype="bfloat16",
                enforce_eager=True,
                limit_mm_per_prompt={
                    "image": 1 
                } # In the CRAG-MM dataset, every conversation has at most 1 image
            )
            print(f"Initializing {self.model_name} with vLLM...")
            self.tokenizer = self.vllm.get_tokenizer()

        print("Models loaded successfully")
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        return 5
    
    def attempt_api_call(
        self,
        client: OpenAI,
        model_name: str,
        messages: list,
        max_retries: int = 3,
    ) -> str | None:
        """
        Attempt a structured output call to the OpenAI API with retries.

        Args:
            client: The OpenAI client instance to use for the API call.
            model_name: The model to query (e.g., "gpt-4o-mini").
            messages: List of message objects for the conversation.
            max_retries: Maximum number of retry attempts before giving up.

        Returns:
            CRAGTurnEvaluationResult object if successful, None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_message = f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed after {max_retries} attempts: {str(e)}[/red]")
                else:
                    console.print(f"[yellow]{error_message}, retrying...[/yellow]")
        return None
    
    def truncate_to_token_limit(self, text, max_tokens):
        if 'gpt' not in self.model_name:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = self.tokenizer.decode(tokens)
        else:
            text = text[:min(len(text), int(max_tokens * 4))] # 100 tokens - 75 words / 4 characters of text
        return text

    def prepare_grounding_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        message_histories: List[List[Dict[str, Any]]],
        filtered_image_search_results_batch: List[List[Dict[str, Any]] | List],
        filtered_text_search_results_batch: List[List[Dict[str, Any]] | List],
    ):
        """Prepare RAG-enhanced inputs using localized image regions."""
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for query, image, message_history, filtered_text_search_results, filtered_image_search_results in zip(
            queries, images, message_histories, filtered_text_search_results_batch, filtered_image_search_results_batch
        ):
            if image:
                SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                            "Keep your response concise and to the point.")
            else:
                SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions."
                            "Keep your response concise and to the point.")
                
            # Add retrieved context if available and search is not disabled
            rag_context = ""
            if filtered_text_search_results:
                snippets = []
                for i, result in enumerate(filtered_text_search_results[:self.text_search_topk]):
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        truncated_snippet = self.truncate_to_token_limit(snippet, 300) # Truncate retrieved documents to handle context length limit
                        snippets.append((i, truncated_snippet, result.get('score')))
                    else:
                        print(f"### DEBUG - no snippet for result {i} {result}")
                
                if snippets:
                    rag_context += "Here is some additional information that may help you answer:\n\n"
                    for i, snippet, score in snippets:
                        rag_context += f"[Info {i+1}] (Score: {score}) {snippet}\n\n"

            if filtered_image_search_results:
                rag_context += "Here is some additional information about the localized object:\n\n"
                for i, result in enumerate(filtered_image_search_results[:self.image_search_topk]):
                    entities = result.get('entities', '')
                    if entities:
                        if not result.get("index"): # GT retrieval info
                            entities = [{
                               "entity_name": ent,
                                "entity_attributes": json.loads(result.get("info",{})).get(ent,{})
                            } for ent in entities]

                        selected_entities_info = [filter_entity_attributes(ent) for ent in entities]
                        entities_str = ',\n'.join([json.dumps(selected_entities_info_ind, indent=2) for selected_entities_info_ind in selected_entities_info])
                        # Limit the string length to 800 characters and add ellipsis if it's longer
                        entities_str = entities_str[:800] + "..." if len(entities_str) > 800 else entities_str
                        rag_context += f"[Object Info {i+1}] {entities_str}\n\n"
                    else:
                        print(f"### DEBUG - no entities for result {i} {result}")
                
            if image:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image"}]}
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            messages.append({"role": "user", "content": query})
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            if images[0]:
                inputs.append({
                    "prompt": formatted_prompt,
                    "multi_modal_data": {
                        "image": image
                    }
                })
            else:
                inputs.append({
                    "prompt": formatted_prompt,
                })

        
        return inputs
    
    def prepare_openai_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        message_histories: List[List[Dict[str, Any]]],
        filtered_image_search_results_batch: List[List[Dict[str, Any]] | List],
        filtered_text_search_results_batch: List[List[Dict[str, Any]] | List],
    ) -> List[List[Dict[str, Any]]]:
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, image_search_results, text_search_results) in enumerate(
            zip(queries, images, message_histories, filtered_image_search_results_batch, filtered_text_search_results_batch)
        ):
            ## removing IDK version
            if image:
                SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                            "Keep your response concise and to the point.")
            else:
                SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions."
                            "Keep your response concise and to the point.")
            
            # Add retrieved context if available
            rag_context = ""
            if text_search_results:
                snippets = []
                rag_context = "Here is some additional information that may help you answer:\n\n"
                for i, result in enumerate(text_search_results):
                    result = WebSearchResult(result)
                    snippet = f"[{result.get('page_name', '')}] {result.get('page_snippet', '')}"
                    if snippet:
                        truncated_snippet = self.truncate_to_token_limit(snippet, 300) # Truncate retrieved documents to handle context length limit
                        snippets.append((i, truncated_snippet, result.get('score')))
                    else:
                        print(f"### DEBUG - no snippet for result {i} {result}")
                if snippets:
                    rag_context += "Here is some additional information that may help you answer:\n\n"
                    for i, snippet, score in snippets:
                        rag_context += f"[Info {i+1}] (Score: {score}) {snippet}\n\n"
            
            rag_context= rag_context[:300*10]

            if image_search_results:
                rag_context += "Here is some additional information about the localized object:\n\n"
                for i, result in enumerate(image_search_results):
                    entities = result.get('entities', '')
                    if entities:
                        if not result.get("index"): # GT retrieval info
                            entities = [{
                               "entity_name": ent,
                                "entity_attributes": json.loads(result.get("info",{})).get(ent,{})
                            } for ent in entities]

                        selected_entities_info = [filter_entity_attributes(ent) for ent in entities]
                        entities_str = ',\n'.join([json.dumps(selected_entities_info_ind, indent=2) for selected_entities_info_ind in selected_entities_info])
                        # Limit the string length to 1000 characters and add ellipsis if it's longer
                        # entities_str = entities_str[:1000] + "..." if len(entities_str) > 1000 else entities_str
                        entities_str = entities_str[:800] + "..." if len(entities_str) > 800 else entities_str
                        rag_context += f"[Object Info {i+1}] (Score: {result.get('score')}) {entities_str}\n\n"
                    else:
                        print(f"### DEBUG - no entities for result {i} {result}") 

            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            if image:
                # Add the current query
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")  # Save image to buffer in JPEG format
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"{query}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                )
                inputs.append(messages)
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"{query}"},
                        ],
                    }
                )
                inputs.append(messages)

        
        return inputs
    
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
        batch_ir_clip :List[List[Dict[str, Any]]],
        batch_ir_clip_GDINORegion:List[List[Dict[str, Any]]],
        batch_ir_clip_GTRegion :List[List[Dict[str, Any]]],
        batch_tr_bge :List[List[Dict[str, Any]]],
        batch_tr_bge_w_clip :List[List[Dict[str, Any]]],
        batch_tr_bge_w_clip_gdino_region:List[List[Dict[str, Any]]],
        batch_tr_bge_w_clip_gt_region :List[List[Dict[str, Any]]],
        batch_tr_bge_w_textual_only_ver_query:List[List[Dict[str, Any]]],
        textual_only_queries: List[str | None],
    ) -> AgentResponse:
        
        #######################
        #### Options for Image retrieval Augmentation
        #######################
        RAG_COMBINATIONS = {   
            # Image Ret. Only Augmentation
            "image_only" : {
                "batch_filtered_image_search_results": batch_ir_clip,
                "batch_filtered_text_search_results": [[] for _ in range(len(queries))]
            },
            "dino_cropped_image_only":{
                "batch_filtered_image_search_results": batch_ir_clip_GDINORegion,
                "batch_filtered_text_search_results": [[] for _ in range(len(queries))]
            },
            "gt_cropped_image_only":{
                "batch_filtered_image_search_results": batch_ir_clip_GTRegion,
                "batch_filtered_text_search_results": [[] for _ in range(len(queries))]
            },
            # Text Ret. Only Augmentation
            "text_only" : {
                "batch_filtered_image_search_results": [[] for _ in range(len(queries))],
                "batch_filtered_text_search_results": batch_tr_bge
            },
            "text_only_w_textual_only_query" : {
                "batch_filtered_image_search_results": [[] for _ in range(len(queries))],
                "batch_filtered_text_search_results": batch_tr_bge_w_textual_only_ver_query
            },
            # Both Ret. Augmentation
            "both_normal_image_text" : {
                "batch_filtered_image_search_results": batch_ir_clip,
                "batch_filtered_text_search_results": batch_tr_bge_w_clip
            },
            "both_dino_cropped_image_text":{
                "batch_filtered_image_search_results": batch_ir_clip_GDINORegion,
                "batch_filtered_text_search_results": batch_tr_bge_w_clip_gdino_region
            },
            "both_gt_cropped_image_text":{
                "batch_filtered_image_search_results": batch_ir_clip_GTRegion,
                "batch_filtered_text_search_results": batch_tr_bge_w_clip_gt_region
            },   
            # NO RAG
            "no_augmentation": {
                "batch_filtered_image_search_results": [[] for _ in range(len(queries))],
                "batch_filtered_text_search_results": [[] for _ in range(len(queries))]
            },
            "no_augmentation_text_only": {
                "batch_filtered_image_search_results": [[] for _ in range(len(queries))],
                "batch_filtered_text_search_results": [[] for _ in range(len(queries))]
            },
        }

        print(f"##### Selected TYPE: {self.rag_type} for retrieval augmentation!")

        assert self.rag_type in RAG_COMBINATIONS, f"rag-type argument should be within {RAG_COMBINATIONS.keys()}"

        batch_filtered_image_search_results = RAG_COMBINATIONS[self.rag_type]['batch_filtered_image_search_results']
        batch_filtered_text_search_results = RAG_COMBINATIONS[self.rag_type]['batch_filtered_text_search_results']

        if self.rag_type in ["text_only_w_textual_only_query","no_augmentation_text_only"]:
            images = [None] * len(queries)
            queries = textual_only_queries
            print("Use no image for text_only_w_textual_only_query")
            print("textual_only_queries: ",textual_only_queries)
            

        text_search_query_list = [self.rag_type for _ in queries]
        
        if 'gpt' in self.model_name:
            rag_inputs = self.prepare_openai_rag_enhanced_inputs(
                queries, images, message_histories, batch_filtered_image_search_results, batch_filtered_text_search_results
            )
            responses=[]
            for rag_input in rag_inputs:
                response = self.attempt_api_call(self.vllm, self.model_name, rag_input)
                responses.append(response)
        else:
            # Step 1: Prepare RAG-enhanced inputs in batch
            rag_inputs = self.prepare_grounding_rag_enhanced_inputs(
                queries, images, message_histories, batch_filtered_image_search_results, batch_filtered_text_search_results
            )
            
            # Print formatted inputs before API call
            for i, formatted_input in enumerate(rag_inputs):
                print(f"Formatted Input {i+1}:")
                print(formatted_input["prompt"])
                print("-" * 80)
            
            # Step 2: Generate responses using the batch of RAG-enhanced prompts
            print(f"Generating responses for {len(rag_inputs)} queries")
            outputs = self.vllm.generate(
                rag_inputs,
                sampling_params=vllm.SamplingParams(
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=self.max_generation_tokens,
                    skip_special_tokens=True
                )
            )
            
            # Extract and return the generated responses
            responses = [output.outputs[0].text for output in outputs]
        
        print(f"Successfully generated {len(responses)} responses")

        
        return AgentResponse(
            response=responses,
            image_summaries=[None] * len(responses),
            rag_inputs=[None for prompt_dict in rag_inputs], # If you want to save this, change None to prompt_dict['prompt'] 
            text_search_results_batch=batch_filtered_text_search_results,
            image_search_results_batch=batch_filtered_image_search_results,
            text_search_query_list=text_search_query_list,
            intermediate_steps = {
                "object_descriptions": [None] * len(queries),
                "bbox_list": [None] * len(queries),
            },
        )    
