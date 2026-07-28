"""
This file is manipulated and edited based on the code from the following repository: 
https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit
"""
import vllm
import torch
from datasets import load_from_disk, load_dataset
import os 
import argparse
import tqdm
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
from typing import Callable, List
from PIL import Image
from rich.console import Console
import re
import io
import base64
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    OwlViTProcessor, 
    OwlViTForObjectDetection
)

from utils.utils import display_results, evaluate_grounding
from data_loader import DataBatchIterator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

class QAEvaluationResult(BaseModel):
    """Structured output model for QA evaluation results."""
    accuracy: bool
    explanation: str


class ModularEvaluation:
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.model_name = args.model_name
        self.num_workers = 8
        self.max_retries = 3
        self.max_generation_tokens=75
        self.show_progress = True
        self.eval_model_name = args.eval_model
        self.task_type = args.task_type
        self.num_conversations = len(args.dataset) if args.num_conversations is None else min(args.num_conversations, len(self.dataset))
        self.batch_size = args.batch_size
        if 'gpt' not in self.model_name.lower():
            self.load_model()        
        self.openai = OpenAI()

    def load_model(self):
        if 'grounding-dino' in self.model_name:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.grounding_processor = AutoProcessor.from_pretrained(self.model_name)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
            self.grounding_dino_box_threshold = 0.4
            self.grounding_dino_text_threshold = 0.3
        
        elif 'owlvit' in self.model_name:
            self.grounding_processor = OwlViTProcessor.from_pretrained(self.model_name)
            self.grounding_model = OwlViTForObjectDetection.from_pretrained(self.model_name)

        elif 'gpt' not in self.model_name:
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {num_gpus}")
            vllm_tensor_parallel_size=num_gpus
            vllm_gpu_memory_utilization=0.8
            max_model_len = 8192
            max_num_seqs = 2

            self.llm = vllm.LLM(
                self.model_name,
                tensor_parallel_size=vllm_tensor_parallel_size, 
                gpu_memory_utilization=vllm_gpu_memory_utilization, 
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                trust_remote_code=True,
                dtype="bfloat16",
                enforce_eager=True,
                limit_mm_per_prompt={
                    "image": 1 
                },
            )

            self.tokenizer = self.llm.get_tokenizer()

    def batch_generate_response_api(self,queries : List[str | None], images: List[str | None], system_prompt: str)->List[str]:
        prompts = []
        for query,image in zip(queries, images):
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if image is not None:    
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")  # Save image to buffer in JPEG format
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"# Question: \"{query}\"\n"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"# Question: \"{query}\"\n"},
                        ],
                    }
                )

            prompts.append(messages)

        outputs=[]
        for input in prompts:
            completion = self.openai.chat.completions.create(
                    model=self.model_name,
                    messages=input,
            )
            output_text = completion.choices[0].message.content.strip()
            outputs.append(output_text)

        return outputs

    def batch_generate_response(self,queries : List[str | None], images: List[str | None], system_prompt: str)->List[str]:
        if len(queries) != len(images):
            raise ValueError(
                "Each query must have exactly one corresponding image entry: "
                f"got {len(queries)} queries and {len(images)} images."
            )

        prompts = []
        for query, image in zip(queries, images):
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if image is not None:
                messages.append(
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"# Question: \"{query}\"\n"}]}
                )
            else:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": f"# Question: \"{query}\"\n"}]}
                )

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(formatted_prompt)

        
        if ('instructblip' in self.model_name):
            prompts =  self.processor(
                images= [item['image'] for item in prompts],
                text = [item['prompt'] for item in prompts],
                padding=True,
                return_tensors="pt"
            )
            outputs = self.llm.generate(
                    **prompts,
                    temperature=0.1,
                    num_beams=5,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    do_sample=False,
                    max_length=self.max_generation_tokens,
                )
            response_text_list = [output.strip() for output in self.processor.batch_decode(outputs, skip_special_tokens=True)]
            
        else:
            inputs = []
            for prompt, image in zip(prompts, images):
                model_input = {"prompt": prompt}
                if image is not None:
                    model_input["multi_modal_data"] = {"image": image}
                inputs.append(model_input)
            
            outputs = self.llm.generate(
                inputs,
                sampling_params=vllm.SamplingParams(
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=self.max_generation_tokens,
                    skip_special_tokens=True
                )
            )
            response_text_list = [output.outputs[0].text.strip() for output in outputs] 

        return response_text_list

    def batch_generate_coordinates(self,queries: List[str],images:List[Image.Image]) -> List[List[float]]:
        """
        Generate coordinates of target object for the given inputs.

        Args:
            queries: The OpenAI client instance to use for the API call.
            images: The model to query (e.g., "gpt-4o-mini").

        Returns:
            QAEvaluationResult object if successful, None if all attempts fail.
        """
        
        if 'grounding-dino' in self.model_name:
            inputs = self.grounding_processor(
                images=images,
                text=[[desc.replace('.', '').strip()] for desc in queries],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)        
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.grounding_dino_box_threshold,
                text_threshold=self.grounding_dino_text_threshold,
                target_sizes=[img.size[::-1] for img in images]
            )
        
        elif 'owlvit' in self.model_name:
            # NOTE: OWL-VIT expects shared text for all images
            results = []
            for img, query in zip(images, queries):
                processed_text_labels = [[query.replace('.', '').strip()]]
                inputs = self.grounding_processor(
                    text=processed_text_labels,
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)
                
                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                target_sizes = torch.tensor([img.size[::-1]])
                # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                result = self.grounding_processor.post_process_grounded_object_detection(
                    outputs=outputs, 
                    target_sizes=target_sizes, 
                    threshold=0.1, 
                    text_labels=processed_text_labels
                )
                results.append(result[0])
        else:
            raise NotImplementedError("seletected visual grounding model is not implemented.")

        batch_predicted_coordinates = []
        for img, desc, result in zip(images, queries, results):
            if len(result["boxes"]) == 0:
                print(f"No object detected for '{desc}', using original image")
                width, height = img.size
                batch_predicted_coordinates.append([0,0,width,height])
            else:
                best_box = result["boxes"][0].tolist()
                best_score = result["scores"][0]
                best_label = result["labels"][0]
                print(f"Detected '{best_label}' with confidence {best_score:.3f} at location {best_box}")
                predicted_coordinates = list(map(float, best_box))
                batch_predicted_coordinates.append(predicted_coordinates)
                
        return batch_predicted_coordinates
    
    def attempt_api_call(
        self,
        client: OpenAI,
        model_name: str,
        messages: list,
    ) -> QAEvaluationResult | None:
        """
        Attempt a structured output call to the OpenAI API with retries.

        Args:
            client: The OpenAI client instance to use for the API call.
            model_name: The model to query (e.g., "gpt-4o-mini").
            messages: List of message objects for the conversation.

        Returns:
            QAEvaluationResult object if successful, None if all attempts fail.
        """
        for attempt in range(self.max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=QAEvaluationResult,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                error_message = f"API call failed on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                if attempt == self.max_retries - 1:
                    logger.warning(f"[red]Failed after {self.max_retries} attempts: {str(e)}[/red]")
                else:
                    logger.warning(f"[yellow]{error_message}, retrying...[/yellow]")
        return None
    
    def _load_system_prompt(self) -> str:
        """
        get task specific prompt per task_type.
        """
        if self.task_type == "whole":
            system_prompt = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                "Keep your response concise and to the point.")
        elif self.task_type =="visual_grounding":
            system_prompt = (
                "You are a visual grounding assistant. Given an image and a question, output the "
                "bounding box of the image region that contains the visual information needed to "
                "answer the question.\n\n"
                "The image is always 960 pixels wide and 1280 pixels tall.\n"
                "Output format: [x1, y1, x2, y2] in pixels — NO other text.\n"
                "- (0, 0) is the top-left corner\n"
                "- x increases rightward (0 to 960), y increases downward (0 to 1280)\n"
                "- Make the box as tight as possible around the target\n"
                "- If no specific region applies, output the full image: [0, 0, 960, 1280]\n\n"
                "Examples:\n"
                "Question: 'What is the name of the store?'\n"
                "Output: [95, 40, 530, 160]\n\n"
                "Question: 'What color is the car?'\n"
                "Output: [260, 580, 720, 940]\n\n"
                "Question: 'How many windows does the building have?'\n"
                "Output: [60, 180, 900, 1100]\n"
            )
        elif self.task_type =="object_identification":
            system_prompt = (
                "You are a helpful visual assistant that identifies the specific target object in an image referred to by a question. "
                "Your task is not to answer the question, but to determine which object in the image the question is referring to and describe or name that object precisely. "
                "I have highlighted the target object for the question. What is the object’s name which will help me answer the query? "
                "Focus only on visual and contextual cues from the image that indicate the subject of the question.\n\n"
                "Instructions:\n"
                "- Focus on highlighted part of the image with bounding box to answer target object name.\n"
                "- Ignore the question’s semantic intent (e.g., do not explain, justify, or give an opinion-based answer).\n"
                "- Identify the visual target most relevant to the question.\n"
                "- Output only the exact entity object name (e.g., 'Subaru WRX', 'The Empire State Building', 'euphorbia aphylla').\n"
                "- Do not include any additional explanation, reasoning, or answer content.\n\n"
                "Example:\n"
                "Image: A photo of a blue Subaru WRX in a parking lot and the blue Subaru WRX is highlighted with a green border box.\n"
                "Question: “Is this a good car for transporting seven passengers at once?”\n"
                "Correct Output: Subaru WRX\n"
                "Incorrect Output: No, the Subaru WRX can only fit 5 passengers.\n"
                "Now analyze the following image and question to output only the target object name.\n\n"
                "### Response format:\n"
                "target_object: [entity name of target object]\n\n"
            )
        elif self.task_type == "object_identification_easy_query":
            system_prompt = (
                "You are a helpful visual assistant that identifies the specific target object in an image referred to by a question. "
                "Your task is to determine which object in the image the question is referring to and describe or name that object precisely. "
                "I have highlighted the target object for the question. What is the object’s name which will help me answer the query? "
                "Focus only on visual and contextual cues from the image that indicate the subject of the question.\n\n"
                "Instructions:\n"
                "- Focus on highlighted part of the image with bounding box to answer target object name.\n"
                "- Ignore the question’s semantic intent (e.g., do not explain, justify, or give an opinion-based answer).\n"
                "- Identify the visual target most relevant to the question.\n"
                "- Output only the exact entity object name (e.g., 'Subaru WRX', 'The Empire State Building', 'euphorbia aphylla').\n"
                "- Do not include any additional explanation, reasoning, or answer content.\n\n"
                "Example:\n"
                "Image: A photo of a blue Subaru WRX in a parking lot and the blue Subaru WRX is highlighted with a green border box.\n"
                "Question: What is the name of the object?\n"
                "Correct Output: Subaru WRX\n"
                "Incorrect Output: No, the Subaru WRX can only fit 5 passengers.\n"
                "Now analyze the following image and question to output only the target object name.\n\n"
                "### Response format:\n"
                "target_object: [entity name of target object]\n\n"
            )     
        elif self.task_type == "object_identification_with_original":
            system_prompt = (
                "You are a helpful visual assistant that identifies the specific target object in an image referred to by a question. "
                "Your task is not to answer the question, but to determine which object in the image the question is referring to and describe or name that object precisely. "
                "What is the object’s name which will help me answer the query? "
                "Focus only on visual and contextual cues from the image that indicate the subject of the question.\n\n"
                "Instructions:\n"
                "- Ignore the question’s semantic intent (e.g., do not explain, justify, or give an opinion-based answer).\n"
                "- Identify the visual target most relevant to the question.\n"
                "- Output only the exact entity object name (e.g., 'Subaru WRX', 'The Empire State Building', 'euphorbia aphylla').\n"
                "- Do not include any additional explanation, reasoning, or answer content.\n\n"
                "Example:\n"
                "Image: A photo of a blue Subaru WRX in a parking lot and the blue Subaru WRX.\n"
                "Question: “Is this a good car for transporting seven passengers at once?”\n"
                "Correct Output: Subaru WRX\n"
                "Incorrect Output: No, the Subaru WRX can only fit 5 passengers.\n"
                "Now analyze the following image and question to output only the target object name.\n\n"
                "### Response format:\n"
                "target_object: [entity name of target object]\n\n"
            )
        elif self.task_type == "object_identification_image_only":
            system_prompt = (
                "You are a helpful visual assistant that identifies the specific target object in an image referred to by a question. "
                "Your task is to determine which object in the image the question is referring to and describe or name that object precisely. "
                "What is the object’s name which will help me answer the query? "
                "Focus only on visual and contextual cues from the image that indicate the subject of the question.\n\n"
                "Instructions:\n"
                "- Ignore the question’s semantic intent (e.g., do not explain, justify, or give an opinion-based answer).\n"
                "- Identify the visual target most relevant to the question.\n"
                "- Output only the exact entity object name (e.g., 'Subaru WRX', 'The Empire State Building', 'euphorbia aphylla').\n"
                "- Do not include any additional explanation, reasoning, or answer content.\n\n"
                "Example:\n"
                "Image: A photo of a blue Subaru WRX in a parking lot and the blue Subaru WRX is highlighted with a green border box.\n"
                "Question: What is the name of the object?\n"
                "Correct Output: Subaru WRX\n"
                "Incorrect Output: No, the Subaru WRX can only fit 5 passengers.\n"
                "Now analyze the following image and question to output only the target object name.\n\n"
                "### Response format:\n"
                "target_object: [entity name of target object]\n\n"
            )      
        elif self.task_type =="knowledge_extraction":
            system_prompt = ("You are a helpful assistant that truthfully answers user questions."
                "Keep your response concise and to the point."
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        return system_prompt

    def get_evaluation_system_prompt(self) -> str:
        """
        Returns the system message for the evaluator.
        """
        if self.task_type in ["whole", "knowledge_extraction"]:
            return (
                "You will be given a question, a ground truth answer, and a model prediction. "
                "Your task is to judge if the prediction is correct or not based on the ground truth answer.\n\n"
                "## Instructions\n"
                "Read the question, ground truth answer, and model prediction carefully. Follow the step by step guideline below to make a judgment.\n\n"
                "1. If the prediction indicates uncertainty or refusal to answer, output json {'accuracy': False}\n"
                "2. If the prediction exactly matches the ground truth, output json {'accuracy': True}\n"
                "3. If the ground truth is a number\n"
                "\t3.1 If the prediction gives a number that almost exactly matches the ground truth, output json {'accuracy': True}\n"
                "\t3.2 If the prediction gives a number that is not the same as the ground truth, output json {'accuracy': False}\n"
                "4. If the prediction is self-contradictory, output json {'accuracy': False}\n"
                "5. If the prediction is not answering the question, output json {'accuracy': False}\n"
                "6. If ground truth contains a set of objects,\n"
                "\t6.1 if the prediction contains exactly same objects as the ground truth, output json {'accuracy': True}\n"
                "\t6.2 if the prediction contains different objects from the ground truth, output json {'accuracy': False}\n"
                "\t6.3 if the prediction is almost same as the ground truth, use your best judgement to give output.\n"
                "7. If the prediction is grounded by the ground truth, output json {'accuracy': True}\n"
                "8. If the prediction is unrelated or contradictory to the ground truth, output json {'accuracy': False}\n\n"
                "## Additional Guidelines\n"
                "- Take it as granted that the ground truth is always correct.\n"
                "- If the prediction gives extra information that is not in the ground truth, it is still correct as long as it is grounded by the ground truth.\n"
                "- Be careful about numbers. 1 mile is about 1.60934 km. 1 foot is about 0.3048 m. 1 inch is about 2.54 cm. 1 yard is about 0.9144 m. 1 pound is about 0.453592 kg. 1 gallon is about 3.78541 liters. 1 ounce is about 28.3495 grams.\n\n"
                "## Output Format\n"
                "Your judgment should first provide a VERY-SHORT explanation on your rationale. When relevant, you need to include the guidelines above to explain your judgment. \n"
                "Finally, your judgment should clearly state \"answer: True\" or \"answer: False\".\n"
                "Below are some examples:\n"
                "EXAMPLES START\n"
                "Question: who will win the game?\n"
                "Ground Truth: Lakers is favored to win the game.\n"
                "Prediction: Sorry, it is hard to predict the outcome of the game.\n"
                """
                {
                    'explanation': 'The prediction indicates it is not sure about the answer. So the prediction is incorrect according to the guideline 1.', 
                    'accuracy': False
                }\n
                """
                " . . ."
                "EXAMPLES END"
            )
        elif self.task_type in ["object_identification","object_identification_easy_query", "object_identification_with_original","object_identification_image_only"]:
            return (
                "You will be given a ground truth answer and a model prediction. "
                "Your task is to judge if the prediction is correct or not based on the ground truth answer.\n\n"
                "## Instructions\n"
                "Read the ground truth answer and model prediction carefully. Follow the step by step guideline below to make a judgment.\n\n"
                "1. If the prediction indicates uncertainty or refusal to answer, output json {'accuracy': False}\n"
                "2. If the prediction exactly matches the ground truth, output json {'accuracy': True}\n"
                "3. If the prediction exactly matches the ground truth, output json {'accuracy': True}\n"
                "4. If the prediction is different from the ground truth, but is different in a way that is semantically equivalent (synonym), output json {'accuracy': True}\n"
                "\t4.1 if the prediction contains exactly same objects as the ground truth, output json {'accuracy': True}\n"
                "\t4.2 if the prediction contains different objects from the ground truth, output json {'accuracy': False}\n"
                "\t4.3 if the prediction is almost same as the ground truth, use your best judgement to give output.\n"
                "## Output Format\n"
                "Your judgment should first provide a VERY-SHORT explanation on your rationale. When relevant, you need to include the guidelines above to explain your judgment. \n"
                "Finally, your judgment should clearly state \"answer: True\" or \"answer: False\".\n"
                "Below are some examples:\n"
                "EXAMPLES START\n"
                "Ground Truth: KIA K5\n"
                "Prediction: KIA Optima\n"
                """
                {
                    'explanation': 'The prediction indicates different name with the ground truth, but KIA K5 is also formerly known as the Kia Optima. So the prediction is correct according to the guideline 3.', 
                    'accuracy': True
                }\n
                """
                "Ground Truth: Ford F-150\n"
                "Prediction: Ford F-series\n"
                """
                {
                    'explanation': 'The prediction indicates a broader category than the ground truth, and it is not semantically equivalent. So the prediction is incorrect.', 
                    'accuracy': False
                }\n
                """
                " . . ."
                "EXAMPLES END"
            )
    
    def _parse_response(self,response_list : List[str], images: List[Image.Image]) -> List[List[any]]:
        batch_output_string = []
        for response_text, image in zip(response_list, images):
            if self.task_type == "visual_grounding":
                print("DEBUG - target grounding: ",response_text)
                match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
                if match:
                    pred_bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                    batch_output_string.append(pred_bbox)
                else:
                    print(f"Error parsing target region: No bbox found in {response_text}")
                    width, height = image.size
                    batch_output_string.append([0,0,width,height])

            elif self.task_type in ["object_identification","object_identification_easy_query", "object_identification_with_original","object_identification_image_only"]:
                if response_text.startswith('[') and response_text.endswith(']'):
                    response_text = response_text[1:-1]
                if '[' in response_text:
                    st_idx = response_text.index('[')
                    ed_idx = response_text.index(']')
                    response_text= response_text[st_idx+1:ed_idx]
                if 'target_object:' in response_text:
                    st_idx = response_text.index(':')
                    response_text = response_text[st_idx+1:].strip()
                batch_output_string.append(response_text)
            elif self.task_type in ["whole", "knowledge_extraction"]:
                batch_output_string.append(response_text)

        return batch_output_string

    def _evaluate_response(self, target_example):
        pred = target_example["agent_response"]
        ground_truth = target_example["ground_truth"]
        query = target_example["query"]

        is_semantically_correct = False
        is_correct = False
        api_response = None

        # Begin by assuming exact match correctness
        if self.task_type =="visual_grounding":
            is_exact_match = evaluate_grounding(pred, ground_truth, iou_threshold=0.5)
            is_correct = is_exact_match
        elif self.task_type in ["object_identification","object_identification_easy_query", "object_identification_with_original","object_identification_image_only"]:
            is_exact_match = pred.strip().lower() == ground_truth.strip().lower()
            
            if is_exact_match:
                is_correct = is_exact_match
            elif self.eval_model_name:
                # Cover semantically correct cases via LLM-as-a-judge.
                messages = [
                    {"role": "system", "content": self.get_evaluation_system_prompt()},
                    {"role": "user", "content": f"Ground truth: {ground_truth}\nPrediction: {pred}\n"},
                ]
                
                api_response = self.attempt_api_call(self.openai, self.eval_model_name, messages)
                if api_response:
                    is_semantically_correct = api_response.accuracy
                    is_correct = is_semantically_correct

        elif self.task_type in ["whole","knowledge_extraction"]:
            is_exact_match = pred.strip().lower() == ground_truth.strip().lower()
        
            if is_exact_match:
                is_correct = is_exact_match
            elif self.eval_model_name:
                messages = [
                    {"role": "system", "content": self.get_evaluation_system_prompt()},
                    {"role": "user", "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {pred}\n"},
                ]    
                api_response = self.attempt_api_call(self.openai, self.eval_model_name, messages)
                if api_response:
                    is_semantically_correct = api_response.accuracy
                    is_correct = is_semantically_correct
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        if is_exact_match:
            is_semantically_correct = True
        
        return {
            **target_example,
            "task_type": self.task_type,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response.model_dump() if api_response else None,
        }

    def evaluate_responses(
        self,
        turn_data: list[dict[str, any]],
        progress_callback: Callable[[int, int], None] = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
        """
        Phase 2: Evaluate agent responses and calculate scores.

        This method uses a thread-based parallel executor to avoid pickling issues.
        Args:
            turn_data: List of turn data including agent responses.
        Returns:
            A tuple containing turn evaluation results and score dictionaries.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._evaluate_response, data) for data in turn_data]
            for future_idx, future in tqdm.tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Evaluating responses", disable=not self.show_progress):
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(future_idx, len(turn_data))

        # Convert the interim evaluation results to a pandas dataframe
        turn_evaluation_results_df = pd.DataFrame(results)
        turn_evaluation_results_df = turn_evaluation_results_df.sort_values(by=["session_id"])

        if 'referring_exp_label' in turn_evaluation_results_df:
            ambig_turn_evaluation_results_df = turn_evaluation_results_df[turn_evaluation_results_df["referring_exp_label"] == "not enough cue (ambiguous)"]
        else:
            ambig_turn_evaluation_results_df= turn_evaluation_results_df

        all_scores_dictionary = self.calculate_scores(turn_evaluation_results_df)
        ambig_scores_dictionary = self.calculate_scores(ambig_turn_evaluation_results_df)

        turn_evaluation_results = {"all": turn_evaluation_results_df, "ambig": ambig_turn_evaluation_results_df}
        score_dictionaries = {"all": all_scores_dictionary, "ambig": ambig_scores_dictionary}

        return turn_evaluation_results, score_dictionaries

    def calculate_scores(self, turn_evaluation_results_df: pd.DataFrame) -> dict[str, float]:
        """
        Calculate scores for both single-turn and multi-turn conversations.

        Args:
            turn_evaluation_results_df: DataFrame with evaluation results for turns.
        Returns:
            Dictionary of calculated metrics.
        """
        def _set_is_correct_false_after_consecutive(group: pd.DataFrame) -> pd.DataFrame:
            """
            Mark as is_miss after consecutive incorrect responses
            and calculate multi-turn conversation score for each conversation.
            """
            group_copy = group.copy().reset_index(drop=True)
            for i in range(1, len(group_copy)):
                if not group_copy.loc[i - 1, 'is_correct'] and not group_copy.loc[i, 'is_correct']:
                    group_copy.loc[i + 1:, 'is_correct'] = False
                    group_copy.loc[i + 1:, 'is_exact_match'] = False
                    group_copy.loc[i + 1:, 'is_semantically_correct'] = False
                    break

            return group_copy

        turn_evaluation_results_df = turn_evaluation_results_df.groupby("session_id", group_keys=False)[turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

        total = len(turn_evaluation_results_df)
        correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
        correct_semantic = turn_evaluation_results_df["is_semantically_correct"].sum()
        correct = turn_evaluation_results_df["is_correct"].sum()
        
        exact_match = correct_exact / total
        accuracy = correct / total
        
        scores_dictionary = {
            "total": float(total),
            "correct_exact": float(correct_exact),
            "correct_semantic": float(correct_semantic),
            "correct": float(correct),
            "exact_match": float(exact_match),
            "accuracy": float(accuracy),
        }

        return scores_dictionary

    def initialize_evaluation(self) -> None:
        """
        Initialize variables needed for agent evaluation.

        This method sets internal state including the batch iterator, conversation count, 
        agent response map, and turn data list.
        """
        logger.info(f"[blue]Starting evaluation with {self.num_workers} workers[/blue]")
        if self.eval_model_name:
            logger.info(f"[blue]Using semantic evaluation with model: {self.eval_model_name}[/blue]")

        self.conversations_count = len(self.dataset) if self.num_conversations is None else min(self.num_conversations, len(self.dataset))
        batch_size = self.batch_size
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        self.batch_iterator = DataBatchIterator(dataset=self.dataset, batch_size=batch_size, task_type = self.task_type, shuffle=False)
        
    def generate_agent_responses(self, progress_callback: Callable[[int, int], None] = None, args=None) -> None:
        """
        Phase 1: Generate agent responses for each turn in the dataset.
        Phase 1: Generate agent responses for each turn in the dataset.

        This method iterates over the dataset batches using the internal batch iterator and updates the evaluator's state
        with agent responses and turn data.
        """
        if self.batch_iterator is None:
            raise ValueError("Batch iterator is not initialized. Please call initialize_evaluation() first.")

        for batch_idx, batch in enumerate(tqdm.tqdm(self.batch_iterator, desc="Generating responses", disable=not self.show_progress)):
            interaction_ids = batch["session_ids"]
            queries = batch["queries"]
            images = batch["images"]
            # Generate responses for the current batch
            if ('grounding-dino' in self.model_name) or ('owlvit' in self.model_name):
                parsed_response = self.batch_generate_coordinates(queries, images)
            else:
                if 'gpt' not in self.model_name:
                    try:
                        batch_response = self.batch_generate_response(
                            queries,
                            images,
                            system_prompt=self._load_system_prompt(),
                        )
                    except Exception as exc:
                        image_states = [
                            "none" if image is None else f"{image.mode}:{image.size}"
                            for image in images
                        ]
                        raise RuntimeError(
                            "vLLM generation failed for "
                            f"batch {batch_idx}, session_ids={interaction_ids}, "
                            f"images={image_states}"
                        ) from exc
                else:
                    batch_response = self.batch_generate_response_api(queries, images, system_prompt=self._load_system_prompt())
                
                parsed_response = self._parse_response(batch_response, images)
            # assert isinstance(batch_response,AgentResponse)
            # agent_responses = self.truncate_agent_responses(batch_response.response) # Truncase each response to the maximum allowed length (75 tokens)
            # Collect responses and add evaluation data
            for idx, interaction_id in enumerate(interaction_ids):
                agent_response = parsed_response[idx]
                self.agent_response_map[interaction_id] = agent_response
                self.all_turn_data.append({
                    "session_id": batch["session_ids"][idx],
                    "query": queries[idx],
                    "ground_truth": batch["answers"][idx],
                    "agent_response": agent_response,
                    "referring_exp_label": batch['referringe_expression_labels'][idx],
                })
                self.session_ids_evaluated.add(batch["session_ids"][idx])

            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                logger.warning(f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]")
                break

    def evaluate_agent(self, args) -> tuple[dict[str, any], dict[str, any], bool]:
        """
        Evaluate an agent on a dataset and return performance metrics.

        Returns:
            A tuple containing a dictionary of turn evaluation results and a dictionary of scores.
        """
        # Phase 0: Initialize evaluation state
        self.initialize_evaluation()
        
        # Phase 1: Generate agent responses (updates internal state)
        def _generation_progress_callback(conversations_evaluated: int, total_conversations: int) -> None:
            # Can be useful to track progress of the evaluation
            # logging(f"[blue]Generated responses for {conversations_evaluated}/{total_conversations} conversations[/blue]")
            pass

        flag_re_generate = False
        if os.path.exists(os.path.join(args.output_dir, "prediction_result.json")):
            console.print(f"[green]Loading existing predictions from {os.path.join(args.output_dir, 'prediction_result.json')}[/green]")
            with open(os.path.join(args.output_dir, "prediction_result.json"), "r") as f:
                self.all_turn_data = json.load(f)
            flag_re_generate = True
        else:
            self.generate_agent_responses(_generation_progress_callback, args)

            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "prediction_result.json"), "w") as f:
                json.dump(self.all_turn_data, f, indent=2)

        # Phase 2: Evaluate responses using stored turn data
        def _evaluation_progress_callback(turn_evaluated: int, total_turns: int) -> None:
            # Can be useful to track progress of the evaluation
            # logging(f"[blue]Evaluated {turn_evaluated}/{total_turns} turns[/blue]")
            pass
            
        turn_evaluation_results, score_dictionaries = self.evaluate_responses(self.all_turn_data, _evaluation_progress_callback)

        os.makedirs(args.output_dir, exist_ok=True)
        turn_evaluation_results["all"].to_csv(os.path.join(args.output_dir, "turn_evaluation_results_all.csv" if not flag_re_generate else "turn_evaluation_results_all_re_generated_AFTER_SANITY_0128.csv"), index=False)
        turn_evaluation_results["ambig"].to_csv(os.path.join(args.output_dir, "turn_evaluation_results_ambig.csv" if not flag_re_generate else "turn_evaluation_results_ambig_re_generated_AFTER_SANITY_0128.csv"), index=False)
        
        with open(os.path.join(args.output_dir, "scores_dictionary.json" if not flag_re_generate else "scores_dictionary_re_generated.json"), "w") as f:
            json.dump(score_dictionaries, f, indent=2)

        return turn_evaluation_results, score_dictionaries, flag_re_generate
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular intermediate steps Evaluation")
    parser.add_argument(
        "--model-name", 
        type=str,
        required=True, help="VLM model name"
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="OpenAI model for semantic evaluation. Pass 'None' to disable semantic evaluation.",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of conversations to evaluate (default: -1). -1 evaluates all conversations, while a positive number evaluates that many conversations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of batch size for model generation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use ('test')",
    )
    parser.add_argument("--source-dataset-path", type=str, required=True, help="Path to the source dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to save turn evaluation results and scores dictionary",
    )
    parser.add_argument("--task-type", choices=[
        "visual_grounding", 
        "object_identification", 
        "object_identification_easy_query", 
        "object_identification_with_original", 
        "object_identification_image_only",
        "knowledge_extraction", 
        "whole", 
        ], help="Which mode to use.")
    parser.add_argument(
        "--use-ambiguous-label-only",
        action="store_true",
        help="Evaluation only with the ambiguous classified label.",
    )
    args = parser.parse_args()

    if args.eval_model.lower() == "none":
        args.eval_model = None
        console.print(
            "[bold red]WARNING: SEMANTIC EVALUATION IS DISABLED[/bold red]\n"
            "No calls to LLM-as-a-Judge will be made!"
        )

    console.print(
        f"[bold green]Loading dataset from {args.source_dataset_path} [/bold green] "
    )
    
    dataset = load_dataset(args.source_dataset_path)[args.split]

    if args.num_conversations == -1:
        args.num_conversations = len(dataset)
        
    if args.use_ambiguous_label_only:
        # Only select ambiguous label of instances
        dataset = dataset.filter(
            lambda ex: ex['metadata']['referring_expression_category']=="not enough cue (ambiguous)"
        )
        # update question with disambiguated_question
        dataset = dataset.map(
            lambda x: {
                **x,
                "question": x["metadata"]["disambiguated_question"]
            }
        )

    modularEvaluator = ModularEvaluation(args, dataset)
    turn_evaluation_results, score_dictionaries, flag_re_generate = modularEvaluator.evaluate_agent(args)
    display_results(
        console,
        turn_evaluation_results["all"],
        score_dictionaries["all"],
        display_conversations=5,
    )
