"""
This file is manipulated and edited based on the code from the following repository: 
https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit
"""
from datasets import Dataset
import random
from PIL import Image
import cv2 
import numpy as np
from typing import Dict, Any, List
import pickle
import os

from utils.utils import download_image_url, convert_bbox_to_resized_format

class ImageLoader:
    """Handles loading and caching of images from various sources."""

    @staticmethod
    def load_image(conversation_data: Dict[str, Any]) -> Image.Image:
        """
        Load image from conversation data, downloading if necessary.

        Args:
            conversation_data: Dictionary containing image data or URL

        Returns:
            PIL Image object

        Notes:
            - Either 'image' or 'image_url' will be provided in the dataset
            - When the actual image cannot be included, only the image_url is available
        """
        image = conversation_data.get("image")
        image_url = conversation_data.get("image_url")

        if image is None and image_url:
            # Download image from URL (with local caching)
            image_local_path = download_image_url(image_url)
            image = Image.open(image_local_path)

        return image


class DataBatchIterator:
    def __init__(self, dataset: Dataset, batch_size: int, task_type : str, shuffle: bool = False):
        """
        Initialize the batcher with dataset and parameters.

        Args:
            dataset: HuggingFace dataset containing conversations
            batch_size: Number of conversation turns to include in each batch
            task_type : Type of task (e.g., "whole", "visual_grounding", "object_identification", "knowledge_extraction", "object_identification_easy_query", "object_identification_with_original")
            shuffle: Whether to shuffle the conversation order
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.task_type = task_type

        if self.shuffle:
            random.shuffle(self.indices)

    def _extract_turn_data(
        self,
        turn: dict[str, any],
        session_id: str,
        image: Image.Image,
    ) -> dict[str, any]:
        # Extract basic turn information
        interaction_id = turn["id"]
        query = turn["question"]
        answer = turn["answer"]
        referringe_expression_label = turn['metadata']['referring_expression_category']
        assert answer, f"No answer found for interaction_id: {interaction_id}"

        # Resize all image frames to prevent context length limit follwing CRAG-MM.
        image = image.resize((960, 1280))

        # Return structured turn data per task type.
        if self.task_type in ["whole"]:
            return {
                "session_id": session_id,
                "image": image,
                "query": query,
                "answer": answer,
                "referringe_expression_label": referringe_expression_label,
            }
        
        elif self.task_type in ["visual_grounding", "object_identification","object_identification_easy_query", "object_identification_with_original", "object_identification_image_only"]:
            orig_bbox = turn['metadata']['target_roi_bbox']  # [x1, y1, x2, y2]
            orig_size = turn['metadata']['image_size']  # (width, height)
            # All experiments are conducted with the same resize image input to prevent OOM error. We follow experimental setup from CRAG-MM.
            resized_bbox = convert_bbox_to_resized_format(orig_bbox, orig_size, (960, 1280))

            if self.task_type == "visual_grounding":
                return {
                    "session_id": session_id,
                    "image": image,
                    "query": query,
                    "answer": resized_bbox,
                    "referringe_expression_label": referringe_expression_label,
                }
            else:
                if self.task_type in ["object_identification", "object_identification_easy_query"]:
                    box_thickness = 8
                    image_annot = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    x1, y1, x2, y2 = map(int, resized_bbox)
                    boxed_image = cv2.rectangle(image_annot, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
                    # convert to PIL image
                    boxed_image_pil =  Image.fromarray(cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB))

                    print("DEBUG: using cropped image for ablation!!!!")
                    boxed_image_pil = image.crop(tuple(map(float, resized_bbox)))

                    if self.task_type == "object_identification":
                        query = query
                    else:
                        query = "What is the name of the object?"
                elif self.task_type in ["object_identification_with_original","object_identification_image_only"]:
                    boxed_image_pil = image
                    if self.task_type == "object_identification_with_original":
                        query = query
                    elif self.task_type =="object_identification_image_only":
                        query = "What is the name of the object?"
                    query = query 
                else:
                    raise NotImplementedError

                return {
                    "session_id": session_id,
                    "image": boxed_image_pil,
                    "query": query,
                    "answer": turn['metadata']['target_object_name'],
                    "referringe_expression_label": referringe_expression_label,
                }
        
        elif self.task_type in ["knowledge_extraction"]:
            return {
                "session_id": session_id,
                "image": None,
                "query": turn['metadata'].get("textual_only_question", turn["question"]),
                "answer": answer,
                "referringe_expression_label": referringe_expression_label,
            }
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        

    def _collate_batch(self, batch: list[dict[str, any]]) -> dict[str, list[any]]:
        """
        Collate individual turn data into batch format.

        Args:
            batch: List of dictionaries with turn data

        Returns:
            Dictionary with batched data
        """
        # Initialize lists for all fields
        batch_data = {
            "session_ids": [],
            "images": [],
            "queries": [],
            "answers": [],
            "referringe_expression_labels":[],
        }

        # Collect data from each item
        for item in batch:
            batch_data["session_ids"].append(item["session_id"])
            batch_data["images"].append(item["image"])
            batch_data["queries"].append(item["query"])
            batch_data["answers"].append(item["answer"])
            batch_data["referringe_expression_labels"].append(item["referringe_expression_label"])

        return batch_data

    def __iter__(self):
        """
        Iterate through the dataset and yield batches of turns, ensuring that
        turn N+1 for any conversation is only in a strictly later batch
        than turn N from the same conversation.

        This is critical for the correct batching strategy for the multi-turn conversations,
        but should also work for the single-turn conversations.
        """
        from collections import deque

        # For each conversation, track the next turn to produce:
        next_turn_idx = [0] * len(self.dataset)

        # Initialize the queue of conversation indices that have turns left
        queue = deque(self.indices)

        # A cache for conversation data, images, and answers
        self.conversation_cache = {}  # conv_id -> conversation dict
        self.answer_lookup_cache = {}  # conv_id -> {interaction_id -> ans_full}
        self.image_cache = {}  # conv_id -> loaded PIL image

        # We'll accumulate turn data in 'batch' each iteration
        batch = []

        while queue:
            current_convs = []
            # Pop conversation IDs from the queue up to batch_size
            while queue and len(current_convs) < self.batch_size:
                conv_id = queue.popleft()
                current_convs.append(conv_id)

            # Process exactly one turn per conversation in current_convs
            for conv_id in current_convs:
                # ---------------------------
                # 1) LAZY-LOAD IF NECESSARY
                # ---------------------------
                if conv_id not in self.conversation_cache:
                    # Load from dataset once
                    conv_data = self.dataset[conv_id]
                    self.conversation_cache[conv_id] = conv_data

                    # Build answer lookup
                    if isinstance(conv_data, dict):
                        answers = []
                        if isinstance(conv_data["answer"], str):  
                            answers.append(
                                {
                                    "interaction_id": conv_data["id"],
                                    "ans_full": conv_data["answer"],
                                }
                            )
                            conv_data["answers"] = answers    
                        else:
                            assert False, f"Type of conv_data['answers'] is not supported: {type(conv_data['answers'])}"    
                    ans_lookup = {
                        a["interaction_id"]: a["ans_full"] for a in conv_data["answers"]
                    }
                    self.answer_lookup_cache[conv_id] = ans_lookup

                    # Load and cache the image
                    self.image_cache[conv_id] = ImageLoader.load_image(conv_data)

                # -------------------------
                # 2) FETCH FROM THE CACHE
                # -------------------------
                # answer_lookup = self.answer_lookup_cache[conv_id]
                image = self.image_cache[conv_id]
                total_turn_count = 1
                
                # Build the single turn's data
                turn_data = self._extract_turn_data(
                    turn=self.dataset[conv_id],
                    session_id=self.dataset[conv_id]["id"],
                    image=image,
                )
                batch.append(turn_data)

                # ------------------------
                # 3) UPDATE TURN POINTER
                # ------------------------
                next_turn_idx[conv_id] += 1

                # If that conversation still has turns left, re-append it to the (front of the) queue
                if next_turn_idx[conv_id] < total_turn_count:
                    queue.appendleft(conv_id)
                    # note: appending to the left helps keep the cache size in check
                else:
                    # No more turns in this conversation => remove from cache
                    del self.conversation_cache[conv_id]
                    del self.answer_lookup_cache[conv_id]
                    del self.image_cache[conv_id]

            # Yield the entire batch as one chunk
            yield self._collate_batch(batch)
            batch = []

        # If there's anything left in batch, yield it
        if batch:
            yield self._collate_batch(batch)


class PrecomputedDataBatchIterator:
    """
    Processes CRAG-MM-Diagnostic dataset into batches of conversation turns with precomputed information (e.g., Retrieval, Grounding Information).
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False, prebuilt_retrieval_info_path: str | None = None):
        """
        Initialize the batcher with dataset and parameters.

        Args:
            dataset: HuggingFace dataset containing CRAG conversations
            batch_size: Number of conversation turns to include in each batch
            shuffle: Whether to shuffle the conversation order
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prebuilt_retrieval_info_path = prebuilt_retrieval_info_path
        self.indices = list(range(len(dataset)))

        if self.shuffle:
            random.shuffle(self.indices)

        if self.prebuilt_retrieval_info_path is not None and os.path.exists(self.prebuilt_retrieval_info_path):
            with open(self.prebuilt_retrieval_info_path,"rb") as f:
                self.id2_retrieval_info = pickle.load(f)
        else:
            self.id2_retrieval_info = {}
        
        print(f"# Load {len(self.id2_retrieval_info)} id2_retrieval_info_re")
        
    def _extract_turn_data(
        self,
        turn: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Extract basic turn information
        interaction_id = turn["id"]
        query = turn["question"]
        answer = turn["answer"]
        image = turn["image"]
        referringe_expression_label = turn['metadata']['referring_expression_category']
        assert answer, f"No answer found for interaction_id: {interaction_id}"

        retrieved_output = self.id2_retrieval_info.get(interaction_id,{})

        ir_clip = retrieved_output.get("ir_clip",[])
        ir_clip_GDINORegion = retrieved_output.get("ir_clip_GDINORegion",[])
        ir_clip_GTRegion = retrieved_output.get("ir_clip_GTRegion",[])
        tr_bge = retrieved_output.get("tr_bge",[])
        tr_bge_w_clip = retrieved_output.get("tr_bge_w_clip",[])
        tr_bge_w_clip_gdino_region = retrieved_output.get("tr_bge_w_clip_gdino_region",[])
        tr_bge_w_clip_gt_region = retrieved_output.get("tr_bge_w_clip_gt_region",[])
        tr_bge_w_textual_only_ver_query = retrieved_output.get("tr_bge_w_textual_only_ver_query",[])

        vg_gdino_bbox = retrieved_output.get("vg_gdino_bbox",[])
        textual_only_query = turn['metadata'].get("textual_only_question",None)

        orig_bbox = turn['metadata']['target_roi_bbox']  # [x1, y1, x2, y2]
        orig_size = turn['image'].size  # (width, height)
        resized_bbox = convert_bbox_to_resized_format(orig_bbox, orig_size, (960, 1280))

        # Extract conversation history up to this point
        conversation_history = []
        
        # Resize image size following CRAG-MM setup to prevent OOM error.
        image = image.resize((960, 1280))
        
        # Return structured turn data
        return {
            "session_id": interaction_id,
            "image": image,
            "query": query,
            "answer": answer,
            "referringe_expression_label": referringe_expression_label,
            "conversation_history": conversation_history,
            "gt_bounding_box": resized_bbox,
            "vg_gdino_bbox": vg_gdino_bbox,
            "textual_only_query": textual_only_query,
            "ir_clip" :ir_clip,
            "ir_clip_GDINORegion":ir_clip_GDINORegion,
            "ir_clip_GTRegion":ir_clip_GTRegion,
            "tr_bge":tr_bge,
            "tr_bge_w_clip":tr_bge_w_clip,
            "tr_bge_w_clip_gdino_region":tr_bge_w_clip_gdino_region,
            "tr_bge_w_clip_gt_region" :tr_bge_w_clip_gt_region,
            "tr_bge_w_textual_only_ver_query": tr_bge_w_textual_only_ver_query
        }

    def _collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Collate individual turn data into batch format.

        Args:
            batch: List of dictionaries with turn data

        Returns:
            Dictionary with batched data
        """
    
        # Initialize lists for all fields
        batch_data = {
            "session_ids": [],
            "images": [],
            "queries": [],
            "answers": [],
            "referringe_expression_labels":[],
            "conversation_histories": [],
            "gt_bounding_boxes" : [],
            "vg_gdino_bboxes": [],
            "textual_only_queries": [],

            "batch_ir_clip" :[],
            "batch_ir_clip_GDINORegion":[],
            "batch_ir_clip_GTRegion":[],
            "batch_tr_bge":[],
            "batch_tr_bge_w_clip":[],
            "batch_tr_bge_w_clip_gdino_region":[],
            "batch_tr_bge_w_clip_gt_region" :[],
            "batch_tr_bge_w_textual_only_ver_query": []
        }

        # Collect data from each item
        for item in batch:
            batch_data["session_ids"].append(item["session_id"])
            batch_data["images"].append(item["image"])
            batch_data["queries"].append(item["query"])
            batch_data["answers"].append(item["answer"])
            batch_data["referringe_expression_labels"].append(item["referringe_expression_label"])

            batch_data["conversation_histories"].append(item.get("conversation_history",[]))
            batch_data["gt_bounding_boxes"].append(item["gt_bounding_box"])
            
            batch_data["vg_gdino_bboxes"].append(item["vg_gdino_bbox"])
            batch_data["textual_only_queries"].append(item["textual_only_query"])
            batch_data["batch_ir_clip"].append(item.get('ir_clip'))
            batch_data["batch_ir_clip_GDINORegion"].append(item.get('ir_clip_GDINORegion'))
            batch_data["batch_ir_clip_GTRegion"].append(item.get('ir_clip_GTRegion'))
            batch_data["batch_tr_bge"].append(item.get('tr_bge'))
            batch_data["batch_tr_bge_w_clip"].append(item.get('tr_bge_w_clip'))
            batch_data["batch_tr_bge_w_clip_gdino_region"].append(item.get('tr_bge_w_clip_gdino_region'))
            batch_data["batch_tr_bge_w_clip_gt_region"].append(item.get('tr_bge_w_clip_gt_region'))
            batch_data["batch_tr_bge_w_textual_only_ver_query"].append(item.get('tr_bge_w_textual_only_ver_query'))

        return batch_data

    def __iter__(self):
        """
        Iterate through the dataset and yield batches of turns, ensuring that
        turn N+1 for any conversation is only in a strictly later batch
        than turn N from the same conversation.

        This is critical for the correct batching strategy for the multi-turn conversations,
        but should also work for the single-turn conversations.
        """
        from collections import deque

        # For each conversation, track the next turn to produce:
        # next_turn_idx[i] = k means we've consumed turns [0..k-1].
        next_turn_idx = [0] * len(self.dataset)

        # Initialize the queue of conversation indices that have turns left
        queue = deque(self.indices)

        # A cache for conversation data, images, and answers
        self.conversation_cache = {}  # conv_id -> conversation dict
        self.answer_lookup_cache = {}  # conv_id -> {interaction_id -> ans_full}
        self.image_cache = {}  # conv_id -> loaded PIL image

        # We'll accumulate turn data in 'batch' each iteration
        batch = []

        while queue:
            current_convs = []
            # Pop conversation IDs from the queue up to batch_size
            while queue and len(current_convs) < self.batch_size:
                conv_id = queue.popleft()
                current_convs.append(conv_id)

            # Process exactly one turn per conversation in current_convs
            for conv_id in current_convs:
                # ---------------------------
                # 1) LAZY-LOAD IF NECESSARY
                # ---------------------------
                if conv_id not in self.conversation_cache:
                    # Load from dataset once
                    conv_data = self.dataset[conv_id]
                    self.conversation_cache[conv_id] = conv_data

                    # Build answer lookup
                    if isinstance(conv_data, dict):
                        answers = []
                        if isinstance(conv_data["answer"], str):  
                            answers.append(
                                {
                                    "interaction_id": conv_data["id"],
                                    "ans_full": conv_data["answer"],
                                }
                            )
                            conv_data["answers"] = answers    
                        else:
                            assert False, f"Type of conv_data['answers'] is not supported: {type(conv_data['answers'])}"    
                    ans_lookup = {
                        a["interaction_id"]: a["ans_full"] for a in conv_data["answers"]
                    }
                    self.answer_lookup_cache[conv_id] = ans_lookup

                    # Load and cache the image
                    self.image_cache[conv_id] = ImageLoader.load_image(conv_data)

                # -------------------------
                # 2) FETCH FROM THE CACHE
                # -------------------------
                image = self.image_cache[conv_id]

                total_turn_count = 1
                
                # Build the single turn's data
                turn_data = self._extract_turn_data(
                    turn=self.dataset[conv_id],
                )
                batch.append(turn_data)

                # ------------------------
                # 3) UPDATE TURN POINTER
                # ------------------------
                next_turn_idx[conv_id] += 1

                # If that conversation still has turns left, re-append it to the (front of the) queue
                if next_turn_idx[conv_id] < total_turn_count:
                    queue.appendleft(conv_id)
                    # note: appending to the left helps keep the cache size in check
                else:
                    # No more turns in this conversation => remove from cache
                    del self.conversation_cache[conv_id]
                    del self.answer_lookup_cache[conv_id]
                    del self.image_cache[conv_id]

            # Yield the entire batch as one chunk
            yield self._collate_batch(batch)
            batch = []

        # If there's anything left in batch, yield it
        if batch:
            yield self._collate_batch(batch)
