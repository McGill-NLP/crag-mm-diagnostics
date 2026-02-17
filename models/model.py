from typing import Dict, List, Any
from pydantic import BaseModel

class AgentResponse(BaseModel):
    response: List[str | None]
    image_summaries: List[str | None]
    rag_inputs: List[str | None]
    text_search_results_batch : List[List[Dict[str, Any] | None] | None | List] 
    image_search_results_batch : List[List[Dict[str, Any] | None] | None | List] 
    text_search_query_list : List[str | None]
    intermediate_steps : Dict[str,Any] | None

