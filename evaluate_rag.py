"""
This file is manipulated and edited based on the code from the following repository: 
https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from tokenizers import Tokenizer

from models.model import AgentResponse
from models.base_agent import BaseAgent
from models.retriever import Retriever
from models.advancedRAG import AdvancedRAG
from models.advancedRAG_with_precomputed_retrievers import PrecomputedRetrievalAdvancedRAG
from data_loader import PrecomputedDataBatchIterator
from retrievals.search import CustomSearchPipeline
from utils.utils import display_results, ensure_crag_cache_dir_is_configured


# Load environment variables
load_dotenv()
ensure_crag_cache_dir_is_configured()

console = Console()

# Constants for configuration
DEFAULT_EVAL_MODEL = "gpt-4o-mini-2024-07-18"
MAX_API_RETRIES = 3
DEFAULT_NUM_WORKERS = 8

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16


class QAEvaluationResult(BaseModel):
    """Structured output model for CRAG turn evaluation results."""
    explanation : str
    accuracy: bool

class RAGEvaluator:
    """
    A class to evaluate an agent on the CRAG-MM dataset.

    This evaluator generates responses, evaluates them (optionally using a semantic evaluation model),
    computes multi-turn conversation metrics, and (optionally) saves the results.
    """
    def __init__(
        self,
        dataset: Dataset,
        model_type: str,
        agent: BaseAgent,
        eval_model_name: str | None = None,
        num_conversations: int | None = None,
        show_progress: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        prebuilt_retrieval_info_path : str | None = None,
    ) -> None:
        self.dataset = dataset
        self.model_type = model_type
        self.agent = agent
        self.eval_model_name = eval_model_name
        self.num_conversations = num_conversations
        self.show_progress = show_progress
        self.num_workers = num_workers
        self.prebuilt_retrieval_info_path= prebuilt_retrieval_info_path

        # Internal state for evaluation; these are set during initialization
        self.batch_iterator: PrecomputedDataBatchIterator | None = None
        self.conversations_count: int = 0
        self.agent_response_map: dict[str, str] = {}
        self.all_turn_data: list[dict[str, any]] = []
        self.session_ids_evaluated: set[str] = set()
        
        self.tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    @staticmethod
    def get_system_message() -> str:
        """
        Returns the system message for the evaluator.
        """
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

    def attempt_api_call(
        self,
        client: OpenAI,
        model_name: str,
        messages: list,
        max_retries: int = MAX_API_RETRIES,
    ) -> QAEvaluationResult | None:
        """
        Attempt a structured output call to the OpenAI API with retries.

        Args:
            client: The OpenAI client instance to use for the API call.
            model_name: The model to query (e.g., "gpt-4o-mini").
            messages: List of message objects for the conversation.
            max_retries: Maximum number of retry attempts before giving up.

        Returns:
            QAEvaluationResult object if successful, None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=QAEvaluationResult,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                error_message = f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed after {MAX_API_RETRIES} attempts: {str(e)}[/red]")
                else:
                    console.print(f"[yellow]{error_message}, retrying...[/yellow]")
        return None

    def evaluate_response(self, crag_turn_data: dict[str, any]) -> dict[str, any]:
        """
        Evaluate a single response and return evaluation results.

        Args:
            crag_turn_data: A dictionary containing query, ground truth, and agent response.

        Returns:
            A dictionary with evaluation results added to crag_turn_data.
        """
        agent_response = crag_turn_data["agent_response"]
        ground_truth = crag_turn_data["ground_truth"]
        query = crag_turn_data["query"]

        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_semantically_correct = False
        api_response = None

        # Begin by assuming exact match correctness
        is_correct = is_exact_match

        # Use semantic evaluation if not an exact match and an evaluation model is provided.
        if not is_exact_match and self.eval_model_name:
            local_openai_client = OpenAI()
            messages = [
                {"role": "system", "content": self.get_system_message()},
                {"role": "user", "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n"},
            ]
            api_response = self.attempt_api_call(local_openai_client, self.eval_model_name, messages)
            if api_response:
                is_semantically_correct = api_response.accuracy
                is_correct = is_semantically_correct
        if is_exact_match:
            is_semantically_correct = True

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response.model_dump() if api_response else None,
        }

    def initialize_evaluation(self) -> None:
        """
        Initialize variables needed for agent evaluation.

        This method sets internal state including the batch iterator, conversation count, 
        agent response map, and turn data list.
        """
        console.print(f"[blue]Starting evaluation with {self.num_workers} workers[/blue]")
        if self.eval_model_name:
            console.print(f"[blue]Using semantic evaluation with model: {self.eval_model_name}[/blue]")

        self.conversations_count = len(self.dataset) if self.num_conversations is None else min(self.num_conversations, len(self.dataset))
        batch_size = int(np.clip(self.agent.get_batch_size(), MIN_BATCH_SIZE, MAX_BATCH_SIZE))
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        # Instantiate the CRAG turn based batch iterator 
        self.batch_iterator = PrecomputedDataBatchIterator(dataset=self.dataset, batch_size=batch_size, shuffle=False, prebuilt_retrieval_info_path= self.prebuilt_retrieval_info_path)
        
    def generate_agent_responses_dummy(self, progress_callback: Callable[[int, int], None] = None, args=None) -> None:
        for example in self.all_turn_data:
            self.session_ids_evaluated.add(example["session_id"])

        if progress_callback:
            conversations_evaluated = len(self.session_ids_evaluated)
            progress_callback(conversations_evaluated, self.conversations_count)

        if len(self.session_ids_evaluated) > self.conversations_count:
            console.print(f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]")

    def generate_agent_responses(self, progress_callback: Callable[[int, int], None] = None, args=None) -> None:
        """
        Phase 1: Generate agent responses for each turn in the dataset.
        Phase 1: Generate agent responses for each turn in the dataset.

        This method iterates over the dataset batches using the internal batch iterator and updates the evaluator's state
        with agent responses and turn data.
        """
        if self.batch_iterator is None:
            raise ValueError("Batch iterator is not initialized. Please call initialize_evaluation() first.")

        
        pred_save_path= os.path.join(args.output_dir, "prediction_result.jsonl")
        if os.path.exists(pred_save_path):
            console.print(f"[green]Loading existing predictions from {pred_save_path}[/green]")
            with open(pred_save_path, "r") as f:
                self.all_turn_data = [json.loads(l) for l in f]
            console.print(f"[green]Loading existing self.all_turn_data total [{len(self.all_turn_data)}] examples [/green]")
            for example in self.all_turn_data:
                self.session_ids_evaluated.add(example["session_id"])
        
        for batch_idx, batch in enumerate(tqdm.tqdm(self.batch_iterator, desc="Generating responses", disable=not self.show_progress)):
            interaction_ids = batch["session_ids"]

            # Skip if exist:
            if len(self.session_ids_evaluated & set(interaction_ids)):
                console.print(f"[green]SKIP This batch [{batch_idx}] already existing in self.all_turn_data [/green]")
                continue

            queries = batch["queries"]
            images = batch["images"]
            conversation_histories = batch["conversation_histories"]

            # Additional inputs for PrecomputedRetrievalAdvancedRAG
            gt_bounding_boxex = batch["gt_bounding_boxes"]
            textual_only_queries = batch["textual_only_queries"]
            batch_ir_clip= batch["batch_ir_clip"]
            batch_ir_clip_GDINORegion= batch["batch_ir_clip_GDINORegion"] # dino predicted
            batch_ir_clip_GTRegion= batch["batch_ir_clip_GTRegion"]
            batch_tr_bge= batch["batch_tr_bge"]
            batch_tr_bge_w_clip= batch["batch_tr_bge_w_clip"]
            batch_tr_bge_w_clip_gdino_region= batch["batch_tr_bge_w_clip_gdino_region"]
            batch_tr_bge_w_clip_gt_region= batch["batch_tr_bge_w_clip_gt_region"]
            batch_tr_bge_w_textual_only_ver_query= batch["batch_tr_bge_w_textual_only_ver_query"]

            message_histories = []
            interaction_id_histories = []
            # Build message histories for multi-turn conversations
            for conversation_history in conversation_histories:
                message_history = []
                interaction_id_history = []
                for turn in conversation_history:
                    turn_interaction_id = turn["interaction_id"]
                    turn_agent_response = self.agent_response_map.get(turn_interaction_id)
                    if not turn_agent_response:
                        raise AssertionError(
                            f"Agent response not found for turn {turn_interaction_id}. "
                            "Did you shuffle the multi-turn conversations by mistake?"
                        )
                    message_history.append({"role": "user", "content": turn["query"]})
                    message_history.append({"role": "assistant", "content": turn_agent_response})
                    interaction_id_history.append(turn_interaction_id)
                message_histories.append(message_history)
                interaction_id_histories.append(interaction_id_history)

            # Generate responses for the current batch
            if self.model_type =="AdvancedRAG":
                batch_response = self.agent.batch_generate_response(
                    queries, 
                    images, 
                    message_histories,
                )
            elif self.model_type =="PrecomputedRetrievalAdvancedRAG":
                batch_response = self.agent.batch_generate_response(
                    queries, 
                    images, 
                    message_histories, 
                    batch_ir_clip,
                    batch_ir_clip_GDINORegion,
                    batch_ir_clip_GTRegion,
                    batch_tr_bge,
                    batch_tr_bge_w_clip,
                    batch_tr_bge_w_clip_gdino_region,
                    batch_tr_bge_w_clip_gt_region,
                    batch_tr_bge_w_textual_only_ver_query,
                    textual_only_queries
                )
            elif self.model_type=="Retriever":
                batch_response = self.agent.batch_generate_response(
                    queries, 
                    images, 
                    message_histories, 
                    gt_bounding_boxex,
                )
            
            # Handle both old (List[str]) and new (Tuple[List[str], List[float]]) return formats
            assert isinstance(batch_response,AgentResponse)
            
            agent_responses = self.truncate_agent_responses(batch_response.response)
            
            # Collect responses and add evaluation data
            # Save per batch
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "prediction_result.jsonl"), "a") as f:
                for idx, interaction_id in enumerate(interaction_ids):
                    agent_response = agent_responses[idx]
                    self.agent_response_map[interaction_id] = agent_response
                    turn_data={
                        "session_id": batch["session_ids"][idx],
                        "is_ego": None,
                        "referring_exp_label": batch['referringe_expression_labels'][idx],
                        "query": queries[idx],
                        "ground_truth": batch["answers"][idx],
                        "agent_response": agent_response,
                        "debug_image_summaries": batch_response.image_summaries[idx],
                        "debug_rag_inputs": batch_response.rag_inputs[idx],
                        "debug_text_search_results_batch": batch_response.text_search_results_batch[idx],
                        "debug_image_search_results_batch": batch_response.image_search_results_batch[idx],
                        "text_search_query_list": batch_response.text_search_query_list[idx],
                        "intermediate_steps": {k:v[idx] for k,v in batch_response.intermediate_steps.items()} if batch_response.intermediate_steps else None,
                    }
                    self.all_turn_data.append(turn_data)
                    self.session_ids_evaluated.add(batch["session_ids"][idx])

                    f.write(json.dumps(turn_data) + "\n")

            
            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                console.print(f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]")
                break

    def evaluate_agent_responses(
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
            futures = [executor.submit(self.evaluate_response, data) for data in turn_data]
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
        multi_turn_conversation_score_map: dict[str, float] = {}

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
                    group_copy.loc[i + 1:, 'is_miss'] = True
                    group_copy.loc[i + 1:, 'is_semantically_correct'] = False
                    break

            multi_turn_conversation_score = group_copy["is_correct"].mean()
            group_copy["multi_turn_conversation_score"] = multi_turn_conversation_score
            session_id = group_copy.iloc[0]["session_id"]
            multi_turn_conversation_score_map[session_id] = multi_turn_conversation_score
            return group_copy

        turn_evaluation_results_df = turn_evaluation_results_df.groupby("session_id", group_keys=False)[turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

        total = len(turn_evaluation_results_df)
        correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
        correct = turn_evaluation_results_df["is_correct"].sum()
        correct_semantic = turn_evaluation_results_df["is_semantically_correct"].sum()

        exact_match = correct_exact / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        scores_dictionary = {
            "total": float(total),
            "correct_exact": float(correct_exact),
            "correct_semantic": float(correct_semantic),
            "correct": float(correct),
            "exact_match": float(exact_match),
            "accuracy": float(accuracy),
        }

        return scores_dictionary

    def save_results(self, turn_evaluation_results: dict[str, any], scores_dictionary: dict[str, any], output_dir: str,flag_re_generate:bool) -> None:
        """
        Save evaluation results to the specified directory.

        Args:
            turn_evaluation_results: The evaluation results to save.
            scores_dictionary: The scores dictionary to save.
            output_dir: Path where to save the results.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        turn_evaluation_results["all"].to_csv(os.path.join(output_dir, "turn_evaluation_results_all.csv" if not flag_re_generate else "turn_evaluation_results_all_re_generated.csv"), index=False)
        turn_evaluation_results["ambig"].to_csv(os.path.join(output_dir, "turn_evaluation_results_ambig.csv" if not flag_re_generate else "turn_evaluation_results_ambig_re_generated.csv"), index=False)
        with open(os.path.join(output_dir, "scores_dictionary.json" if not flag_re_generate else "scores_dictionary_re_generated.json"), "w") as f:
            json.dump(scores_dictionary, f, indent=2)

    def evaluate_agent(self, args) -> tuple[dict[str, any], dict[str, any],bool]:
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
            # console.log(f"[blue]Generated responses for {conversations_evaluated}/{total_conversations} conversations[/blue]")
            pass
        
        flag_re_generate = False
        predicted_file_path = os.path.join(args.output_dir, "prediction_result.json")
        console.print(f"[green] evaluate_agent :Checking existing predictions file exists from {predicted_file_path}[/green]")
        if os.path.exists(predicted_file_path):
            console.print(f"[green]Loading existing predictions from {predicted_file_path}[/green]")
            with open(predicted_file_path, "r") as f:
                self.all_turn_data = json.load(f)
            console.print(f"[green] Total {len(self.all_turn_data)} self.all_turn_data loaded from the file[/green]")
            flag_re_generate = True
            self.generate_agent_responses_dummy(_generation_progress_callback, args)
        else:
            self.generate_agent_responses(_generation_progress_callback, args)

            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "prediction_result.json"), "w") as f:
                json.dump(self.all_turn_data, f, indent=2)

        # Phase 2: Evaluate responses using stored turn data
        def _evaluation_progress_callback(turn_evaluated: int, total_turns: int) -> None:
            # Can be useful to track progress of the evaluation
            # console.log(f"[blue]Evaluated {turn_evaluated}/{total_turns} turns[/blue]")
            pass
            
        turn_evaluation_results, score_dictionaries = self.evaluate_agent_responses(self.all_turn_data, _evaluation_progress_callback)
        # return turn_evaluation_results, score_dictionaries
        return turn_evaluation_results, score_dictionaries, flag_re_generate
    
    def truncate_agent_responses(self, agent_responses: list[str]) -> list[str]:
        """
        Truncate each agent response to the maximum allowed length.
        """
        encodings = self.tokenizer.encode_batch(agent_responses)
        trimmed_agent_responses = [self.tokenizer.decode(enc.ids) for enc in encodings]
        return trimmed_agent_responses    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on the CRAG-MM dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset type to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use ('test')",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of conversations to evaluate (default: -1). -1 evaluates all conversations, while a positive number evaluates that many conversations.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Suppress search API when calling the agent"
    )
    parser.add_argument(
        "--suppress-web-search-api",
        action="store_true",
        help="Suppress web search API when calling the agent"
    )
    parser.add_argument(
        "--display-conversations",
        type=int,
        default=2,
        help="Number of evaluation examples to show",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="OpenAI model for semantic evaluation. Pass 'None' to disable semantic evaluation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="VLLM Model for Agent Core generator",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to save turn evaluation results and scores dictionary",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of worker processes for parallel evaluation (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="separate",
        choices=["separate", "mllm"],
        help="Search retriever type to use for the agent (default: 'separate'). ",
    )
    parser.add_argument(
        "--retriever-model-name",
        type=str,
        choices=["clip", "Qwen3-VL-Embedding-2B"],
        help='Model name for retriever : (clip-vit-large-patch14-336, Qwen3-VL-Embedding-2B")',
    )
    parser.add_argument(
        "--web-hf-index-id",
        type=str,
        help='HF dataset index id for web search retriever',
    )
    parser.add_argument(
        "--image-hf-index-id",
        type=str,
        help='HF dataset index id for image search retriever',
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["AdvancedRAG", "PrecomputedRetrievalAdvancedRAG", "Retriever"],
        help='Model type to evaluate : ("AdvancedRAG") ',
    )
    parser.add_argument(
        "--prebuilt-retrieval-info-path",
        type=str,
        default=None,
        help="Path to prebuilt retrieval info pickle file (only used for PrecomputedRetrievalAdvancedRAG)",
    )
    parser.add_argument(
        "--use-ambiguous-label-only",
        action="store_true",
        help="Evaluation only with the ambiguous classified label.",
    )
    parser.add_argument(
        "--rag-type",
        type=str,
        required=False,
        choices=[
            "image_only", "dino_cropped_image_only", "gt_cropped_image_only", 
            "text_only","text_only_w_textual_only_query",
            "both_normal_image_text","both_dino_cropped_image_text","both_gt_cropped_image_text",
            "no_augmentation","no_augmentation_text_only"],
        help='RAG type to evaluate for : ("PrecomputedRetrievalAdvancedRAG")',
    )
    
    args = parser.parse_args()

    console.print(f"[bold blue]Loading {args.dataset} dataset...[/bold blue]")
    console.print(
        f"[bold green]Loading dataset from {args.dataset} [/bold green] "
    )
    dataset = load_dataset(args.dataset)[args.split]
    
    console.print(
        f"[bold green]Using split:[/bold green] '{args.split}' with {len(dataset)} examples"
    )

    if args.eval_model.lower() == "none":
        args.eval_model = None
        console.print(
            Panel(
                "[bold red]WARNING: SEMANTIC EVALUATION IS DISABLED[/bold red]\n\n"
                "No calls to LLM-as-a-Judge will be made!\n"
                "Results will rely only on exact string matching.",
                title="[bold red]ATTENTION[/bold red]",
                border_style="red",
                width=100,
                padding=(2, 5),
                expand=False,
            )
        )

    if args.num_conversations == -1:
        args.num_conversations = len(dataset)

    if not args.skip_search and args.model_type !="PrecomputedRetrievalAdvancedRAG":
        if args.retriever_type == "separate":
            # In our experiments, we use default image and text retriever for separate modality type.
            search_api_text_model_name = "BAAI/bge-large-en-v1.5"
            search_api_image_model_name = "openai/clip-vit-large-patch14-336"
            
            search_pipeline = CustomSearchPipeline(
                retriever_type=args.retriever_type,
                text_model_name=search_api_text_model_name,
                image_model_name=search_api_image_model_name,
                web_hf_dataset_id=args.web_hf_index_id if not args.suppress_web_search_api else None,
                image_hf_dataset_id=args.image_hf_index_id
            )
        elif args.retriever_type == "mllm":
            search_pipeline = CustomSearchPipeline(
                retriever_type=args.retriever_type,
                web_hf_dataset_id=None, # We only use image retriever for Qwen3-VL-Embedding-2B in our experiments, so text_index_path is set to None
                image_hf_dataset_id=args.image_hf_index_id,
                multimodal_model_name=args.retriever_model_name,
            )
    else:
        search_pipeline=None
    
    predicted_file_path = os.path.join(args.output_dir, "prediction_result.json")
    console.print(f"[green]Checking existing predictions file exists from {predicted_file_path}[/green]")
    if os.path.exists(predicted_file_path):
        TargetAgent = BaseAgent
    else:
        if args.model_type =="AdvancedRAG":
            TargetAgent = AdvancedRAG
        elif args.model_type =="PrecomputedRetrievalAdvancedRAG":
            TargetAgent = PrecomputedRetrievalAdvancedRAG
            
            assert args.prebuilt_retrieval_info_path is not None, "Prebuilt retrieval info path is required for PrecomputedRetrievalAdvancedRAG"
        elif args.model_type=="Retriever":
            TargetAgent = Retriever
        else:
            raise NotImplementedError
    
    if args.use_ambiguous_label_only:
        dataset = dataset.filter(
            lambda ex: ex['metadata']['referring_expression_category']=="not enough cue (ambiguous)"
        )
        # update question with data['metadata']['disambiguated_question']
        dataset = dataset.map(
            lambda x: {
                **x,
                "question": x["metadata"]["disambiguated_question"]
            }
        )

        console.print(
            f"[bold green]Loading only ambiguous classified label [/bold green] {len(dataset)}"
        )
        console.print(
            f"[bold green]Loading only ambiguous classified label [/bold green] {dataset[0]}"
        )
        
    if args.model_type =="PrecomputedRetrievalAdvancedRAG":
        evaluator = RAGEvaluator(
            dataset= dataset,
            model_type=args.model_type,
            agent=TargetAgent(
                search_pipeline=search_pipeline, 
                model_name=args.model_name,
                rag_type=args.rag_type,
            ),
            eval_model_name=args.eval_model,
            num_conversations=args.num_conversations,
            show_progress=not args.no_progress,
            num_workers=args.num_workers,
            prebuilt_retrieval_info_path=args.prebuilt_retrieval_info_path,
        )
    else:
        evaluator = RAGEvaluator(
        dataset= dataset,
        model_type=args.model_type,
        agent=TargetAgent(
            search_pipeline=search_pipeline, 
            model_name=args.model_name,
        ),
        eval_model_name=args.eval_model,
        num_conversations=args.num_conversations,
        show_progress=not args.no_progress,
        num_workers=args.num_workers,
        prebuilt_retrieval_info_path=args.prebuilt_retrieval_info_path,
    )

    turn_evaluation_results, score_dictionaries,flag_re_generate = evaluator.evaluate_agent(args)

    display_results(
        console,
        turn_evaluation_results["all"],
        score_dictionaries["all"],
        display_conversations=args.display_conversations,
        is_ambig=False,
    )
    if len(turn_evaluation_results["ambig"]) > 0:
        display_results(
            console,
            turn_evaluation_results["ambig"],
            score_dictionaries["ambig"],
            display_conversations=args.display_conversations,
            is_ambig=True,
        )

    if args.output_dir:
        evaluator.save_results(turn_evaluation_results, score_dictionaries, args.output_dir,flag_re_generate)


if __name__ == "__main__":
    main()