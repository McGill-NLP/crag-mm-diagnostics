"""
This file is manipulated and edited based on the code from the following repository: 
https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit
"""
from typing import Any
from utils.utils import maybe_list, lookup_web_content


class BaseRetriever:
    def __init__(self):
        return
    
    def encode_input(self, input):
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self,top_k):
        raise NotImplementedError("Subclasses must implement this method")
    
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
                        **lookup_web_content(self.web_content_private, self.text_web.get_page_url(ind))
                    }
                    for ind, score in results
                ]
            else:
                return [
                    {
                        "index": ind,
                        "score": score,
                        "page_name": self.text_web.get_page_name(ind),
                        "page_snippet": self.text_web.get_page_snippet(ind),
                        "page_url": self.text_web.get_page_url(ind),
                    }
                    for ind, score in results
                ]
        elif model_outputs["type"] == "image":
            return [
                {
                    "index": ind,
                    "score": dist,
                    "url": self.crag_image_kg.get_image_url(ind),
                    "entities": [
                        {
                            "entity_name": entity,
                            "entity_attributes": self.crag_image_kg.get_entity(
                                entity_name=entity
                            ),
                        }
                        for entity in maybe_list(
                            self.crag_image_kg.get_entity_name(ind)
                        )
                    ],
                }
                for ind, dist in results
            ]
        else:
            raise ValueError("Unknown model output type.")