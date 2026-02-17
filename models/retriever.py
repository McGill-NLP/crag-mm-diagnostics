from typing import Dict, List, Any

from PIL import Image
from retrievals.search import CustomSearchPipeline

from rich.console import Console

from .base_agent import BaseAgent
from .model import AgentResponse

console = Console()

class Retriever(BaseAgent):
    def __init__(
        self, 
        search_pipeline: CustomSearchPipeline, 
        model_name=None,
    ):
        super().__init__(search_pipeline)
        self.model_name = model_name
        # Search configuration
        self.activate_image_search=True
        self.activate_text_search=True
        self.use_gt_region=False
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        return 8

    
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        gt_bounding_boxex: List[List[float | int]],
    ) -> AgentResponse:
        print(f"Processing batch of {len(queries)} queries with RAG")
        
        # Batch process search queries
        image_search_results_batch = []
        
        # Using GT grounding box 
        if self.use_gt_region:
            print("Using GT cropped region")
            images = [image.crop(tuple(map(float, gt_bounding_box))) for image, gt_bounding_box in zip(images, gt_bounding_boxex)]
        
        # Retrieve relevant information for each query
        for query, image in zip(queries,images):
            print(f"Using input mode - query: {query} / image: {image}")
            if self.activate_image_search:
                if hasattr(self.search_pipeline.model,"config") and self.search_pipeline.model.config.architectures[0] == "CLIPModel":
                    images_results = self.search_pipeline({"text":None,"image":image}, retrieval_source="image", k=30)      
                else:
                    images_results = self.search_pipeline({"text":query,"image":image}, retrieval_source="image", k=30)      
            
            else:
                images_results= []
            image_search_results_batch.append(images_results)

        text_search_results_batch= []
        text_search_query_list = []
        for i, search_query in enumerate(queries):
            if self.activate_text_search and hasattr(self.search_pipeline,"web_search"):
                text_results = self.search_pipeline({"text":search_query,"image":None}, retrieval_source="text", k=30)      
                text_search_query = search_query
            else:
                text_results=[]
                text_search_query=""
            print(f"Using textual query input {text_search_query}")
            text_search_results_batch.append(text_results)        
            text_search_query_list.append(text_search_query)


        print(f"Successfully generated {len(queries)} responses")
        
        return AgentResponse(
            response=["SANITY"] * len(queries),
            image_summaries=[None] * len(queries),
            rag_inputs=[None] * len(queries),
            text_search_results_batch=text_search_results_batch,
            image_search_results_batch=image_search_results_batch,
            text_search_query_list=text_search_query_list,
            intermediate_steps=None,
        )   

