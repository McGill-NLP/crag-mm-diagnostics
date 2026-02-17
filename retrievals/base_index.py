"""
This file is manipulated and edited based on the code from the following repository: 
https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit
"""
import chromadb
import json
import os
from typing import Any, Union
from huggingface_hub import snapshot_download

class MockWeb(object):
    def __init__(
        self, text_index_path, hf_dataset_id, web_hf_dataset_tag=None
    ):
        if hf_dataset_id:
            try:
                print("Loading web search data from Hugging Face...")

                text_index_path = snapshot_download(
                    repo_id=hf_dataset_id, 
                    repo_type="dataset",
                    revision=web_hf_dataset_tag)
                
            except Exception as e:
                raise RuntimeError(f"Failed to load web search data from Hugging Face: {e}")
        
        client = chromadb.PersistentClient(path=text_index_path)
        self.vector_db = client.get_or_create_collection(name="web_search_embeddings")
        self.index_to_metadata = dict(zip(self.vector_db.get()['ids'], self.vector_db.get(include=["metadatas"])['metadatas']))

    def get_page_name(self, idx):
        # Return the page name for a given index
        return self.index_to_metadata[str(idx)]["page_name"]

    def get_page_snippet(self, idx):
        # Return the page name for a given index
        return self.index_to_metadata[str(idx)]["page_snippet"]

    def get_page_url(self, idx):
        # Return the page URL for a given index
        return self.index_to_metadata[str(idx)]["page_url"]


class ImageKG(object):
    def __init__(
        self, 
        image_index_path: str,
        hf_dataset_id: str,
        image_hf_dataset_tag: str = None
    ):
        if hf_dataset_id is not None:
            print(f"Loading image index from huggingface {hf_dataset_id}")
            image_index_path = snapshot_download(
                repo_id=hf_dataset_id, 
                repo_type="dataset",
                revision=image_hf_dataset_tag
            )
        n_threads = os.cpu_count() or 1
        client = chromadb.PersistentClient(path=image_index_path)
        self.vector_db = client.get_collection(name="image_embeddings")
        # reset number of threads; index will be re-built when querying
        self.vector_db .modify(metadata={"hnsw:num_threads": n_threads})

        batch_size = 1000
        offset = 0
        id2_data = {}

        while True:
            batch = self.vector_db.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset
            )

            if len(batch["ids"]) == 0:
                break

            id2_data.update(
                dict(zip(batch["ids"], batch["metadatas"]))
            )

            offset += batch_size

        self.id2_data = id2_data
        
        self.kg = {}
        for _, data in self.id2_data.items():
            info = json.loads(data['info'])
            for entity in info:
                self.kg[entity] = info[entity]

    def get_image_url(self, image_id: str) -> str:
        return self.id2_data[image_id]["image_url"]

    def get_entity_name(self, image_id: str) -> Union[str, list[str]]:
        return json.loads(self.id2_data[image_id]["entities"])

    def get_entity(self, entity_name: str) -> dict[str, Any]:
        return self.kg[entity_name]
