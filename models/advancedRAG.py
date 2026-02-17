from typing import Dict, List, Any
import json
import torch
from PIL import Image
from openai import OpenAI
import io
import base64

import vllm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from models.base_agent import BaseAgent
from models.model import AgentResponse
from retrievals.search import CustomSearchPipeline
from retrievals.crag_web_result_fetcher import WebSearchResult
from utils.utils import filter_entity_attributes

class AdvancedRAG(BaseAgent):
    def __init__(
        self, 
        search_pipeline: CustomSearchPipeline,
        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ):
        """
        Initialize the AdvancedRAG.
            - In our experiments, we utilize Grounding-DINO as the visual grounding model to localize relevant regions in the image based on the query.
            - Then use these localized regions to perform more targeted image retrieval of relevant information. 
            - We also perform text retrieval using the original query along with the retrieved image information (e.g. named entities from image search results) to retrieve relevant textual information.
            - The retrieved information is then incorporated into the prompt for response generation, allowing for more accurate and contextually relevant answers.
        
        Args:
            search_pipeline: Pipeline for searching relevant information
            model_name: The model name to use for generation
        """
        super().__init__(search_pipeline)
        self.model_name = model_name
        # VLLM configuration
        self.max_gen_len = 64
        # GPU utilization settings 
        self.vllm_tensor_parallel_size=1
        self.vllm_gpu_memory_utilization=0.65
        # These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
        self.max_model_len = 8192
        self.max_num_seqs = 2
        self.max_generation_tokens=75
        ## Search configuration
        self.activate_image_search=True
        self.activate_text_search=True
        self.image_search_topk = 10
        self.text_search_topk = 10
        self.image_search_score_threshold = 0.0
        self.text_search_threshold = 0.0
        self.text_query_reformulation = True
        # Module: Grounding DINO parameters
        self.segment_image_query = True
        self.grounding_dino_model_id = "IDEA-Research/grounding-dino-base"
        self.grounding_dino_box_threshold = 0.4
        self.grounding_dino_text_threshold = 0.3

        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the vLLM model, tokenizer, and Grounding DINO model with appropriate settings.
        """
        print(f"Initializing {self.model_name} with vLLM...")
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
            self.tokenizer = self.vllm.get_tokenizer()

        # Initialize Grounding DINO
        print(f"Initializing {self.grounding_dino_model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grounding_processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(device)
        self.device = device

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
        return 4
    
    def localize_objects_with_grounding_dino_batch(self, images: List[Image.Image], object_descriptions: List[str]) -> tuple[List[Image.Image], List[List]]:
        """
        Batch version: Use Grounding DINO to localize objects in images and return cropped regions.
        Args:
            images: List of input images
            object_descriptions: List of object descriptions to localize
        Returns:
            List[Image.Image]: Cropped images of the localized objects, or original images if no detection
        """
        # Remove periods and strip descriptions
        cleaned_descriptions = [desc.replace('.', '').strip() for desc in object_descriptions]
        # Process all images and texts in batch

        print("Received descriptions for grounding model:\n","\n".join([f"{idx}: {desc}" for idx, desc in enumerate(cleaned_descriptions)]))
        inputs = self.grounding_processor(
            images=images,
            text=[[desc] for desc in cleaned_descriptions],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Post-process detections for each image
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.grounding_dino_box_threshold,
            text_threshold=self.grounding_dino_text_threshold,
            target_sizes=[img.size[::-1] for img in images]
        )

        cropped_images = []
        bbox_list = []
        for img, desc, result in zip(images, cleaned_descriptions, results):
            if len(result["boxes"]) == 0:
                print(f"No object detected for '{desc}', using original image")
                cropped_images.append(img)
                bbox_list.append([])
            else:
                best_box = result["boxes"][0].tolist()
                best_score = result["scores"][0]
                best_label = result["labels"][0]
                print(f"Detected '{best_label}' with confidence {best_score:.3f} at location {best_box}")
                cropped_image = img.crop(tuple(map(float, best_box)))
                cropped_images.append(cropped_image)
                bbox_list.append(best_box)
        return cropped_images, bbox_list
    
    def truncate_to_token_limit(self, text, max_tokens):
        if 'gpt' not in self.model_name:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = self.tokenizer.decode(tokens)
        else:
            text = text[:min(len(text), int(max_tokens * 4))] # 100 tokens - 75 words / 4 characters of text
        return text

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
            string response if successful, None if all attempts fail.
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
                    # console(f"[red]Failed after {max_retries} attempts: {str(e)}[/red]")
                    print(f"\033[92m Failed after {max_retries} attempts: {str(e)}\033[0m")
                # else:
                    # console(f"[yellow]{error_message}, retrying...[/yellow]")
                    print(f"\033[91m {error_message}, retrying...\033[0m")
        return None
    
    def prepare_grounding_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        message_histories: List[List[Dict[str, Any]]],
        image_search_results_batch: List[List[Dict[str, Any]] | List],
        text_search_results_batch: List[List[Dict[str, Any]] | List],
    ):
        """Prepare RAG-enhanced inputs using localized image regions."""
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for query, image, message_history, text_search_results, image_search_results in zip(
            queries, images, message_histories, text_search_results_batch, image_search_results_batch
        ):
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point.")
            
            # Add retrieved context if available and search is not disabled
            rag_context = ""
            if text_search_results:
                snippets = []
                for i, result in enumerate(text_search_results[:self.text_search_topk]):
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

            if image_search_results:
                rag_context += "Here is some additional information about the localized object:\n\n"
                for i, result in enumerate(image_search_results[:self.image_search_topk]):
                    entities = result.get('entities', '')
                    if entities:
                        selected_entities_info = [filter_entity_attributes(ent) for ent in entities]
                        entities_str = ',\n'.join([json.dumps(selected_entities_info_ind, indent=2) for selected_entities_info_ind in selected_entities_info])
                        # Limit the string length to 800 characters and add ellipsis if it's longer
                        entities_str = entities_str[:800] + "..." if len(entities_str) > 800 else entities_str
                        rag_context += f"[Object Info {i+1}] {entities_str}\n\n"
                    else:
                        print(f"### DEBUG - no entities for result {i} {result}")
                
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
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
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs
    
    def prepare_openai_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        message_histories: List[List[Dict[str, Any]]],
        image_search_results_batch: List[List[Dict[str, Any]]], 
        text_search_results_batch: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, image_search_results, text_search_results) in enumerate(
            zip(queries, images, message_histories, image_search_results_batch, text_search_results_batch)
        ):
            # Create system prompt with RAG guidelines
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
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
                    
            
            if image_search_results:
                rag_context += "Here is some additional information about the localized object:\n\n"
                for i, result in enumerate(image_search_results):
                    entities = result.get('entities', '')
                    if entities:
                        selected_entities_info = [filter_entity_attributes(ent) for ent in entities]
                        entities_str = ',\n'.join([json.dumps(selected_entities_info_ind, indent=2) for selected_entities_info_ind in selected_entities_info])
                        # Limit the string length to 800 characters and add ellipsis if it's longer
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
        
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> AgentResponse:
        """
        Generate grounding-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete grounding-based RAG pipeline:
        1. Use Grounding DINO to localize relevant regions
        2. Retrieve relevant information using localized regions
        3. Generate responses incorporating the retrieved context
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
                
        Returns:
            AgentResponse : 
                response: List[str]
                image_summaries: List[str]
                rag_inputs: List[dict]
                text_search_results_batch : List[List[Dict[str, Any] | None]] 
                image_search_results_batch : List[List[Dict[str, Any] | None]]
                text_search_query_list : List[str | None]
                intermediate_steps : Dict[str, Any]
        """
        print(f"Processing batch of {len(queries)} queries with Grounding DINO RAG")
        
        object_descriptions = [None] * len(queries)
        if self.activate_image_search:
            # Step 2: Localize objects using Grounding DINO
            if self.segment_image_query:
                # Use queries as object descriptions to localize relevant regions
                # From the visual grounding intermediate experiments, we found that using the entire query as the object description for grounding reasonable works. Performance may be improved by using a more complex pipeline that extracts noun phrases from the query for Grounding-DINO.
                localized_images, bbox_list = self.localize_objects_with_grounding_dino_batch(images, queries) 
            else:
                localized_images,bbox_list = images, [[] for i in range(len(images))]
            
            # Step 3: Get initial image search results for validation
            batch_image_search_results = []
            # Collect all image search results first
            for query, localized_image in zip(queries, localized_images):
                image_results = self.search_pipeline({"text":None,"image":localized_image}, retrieval_source="image", k=30)
                batch_image_search_results.append(image_results)
        else:
            localized_images,bbox_list = images, [[] for i in range(len(images))]
            batch_image_search_results =  [[] for i in range(len(images))]


        # # Text search
        batch_text_search_results = []
        text_search_query_list = []
        if self.activate_text_search and hasattr(self.search_pipeline,"web_search"):
            # Text search query: query + object description + image search results (named entities)
            for query, image_search_results in zip(queries, batch_image_search_results):
                # Take only entity names from entities in image search results
                entity_names = [entity.get('entity_name', '') for result in image_search_results[:self.image_search_topk] for entity in result.get('entities', [])]
                # Remove duplicate entity names
                entity_names = list(set(entity_names))
                if self.text_query_reformulation:
                    text_search_query = f"{query} ".rstrip(".") + "." + (f" {', '.join(entity_names)}" if entity_names else "")
                else:
                    text_search_query = query
                print(f"text_search_query: {text_search_query}")
                text_search_query_list.append(text_search_query)
                text_results = self.search_pipeline({"text":text_search_query,"image":None}, retrieval_source="text", k=30)
                print(f"Number of text search results: {len(text_results)}")
                print(f"Scores: {', '.join([str(result.get('score', 0.0)) for result in text_results])}")
                batch_text_search_results.append(text_results)
        else:
            text_search_query_list = [None for _ in queries]
            batch_text_search_results = [[] for _ in queries]

        # Step 4: Prepare grounding RAG-enhanced inputs
        if 'gpt' in self.model_name:
            rag_inputs = self.prepare_openai_rag_enhanced_inputs(
                queries, images, message_histories, batch_image_search_results, batch_text_search_results
            )
            print(f"Generating responses for {len(rag_inputs)} queries")

            responses=[]
            for rag_input in rag_inputs:
                response = self.attempt_api_call(self.vllm, self.model_name, rag_input)
                responses.append(response)
        else:
            rag_inputs = self.prepare_grounding_rag_enhanced_inputs(
                queries, images, message_histories, batch_image_search_results, batch_text_search_results
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
            image_summaries=object_descriptions,
            rag_inputs=[None for prompt_dict in rag_inputs], # If you want to save this, change None to prompt_dict['prompt'] 
            text_search_results_batch=batch_text_search_results,
            image_search_results_batch=batch_image_search_results,
            text_search_query_list=text_search_query_list,
            intermediate_steps = {
                "object_descriptions": object_descriptions,
                "bbox_list": bbox_list,
            },
        )    