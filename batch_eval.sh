#!/bin/bash
################################
# Configuration Variables
################################
NUM_SAMPLES=-1  # -1 for all samples
DATASET=McGill-NLP/crag-mm-diagnostic
SPLIT_TYPE=full  # Options: full, disambiguous_only
SPLIT=test

################################
# Task Types
################################
TASK_TYPE_LIST=(
    "visual_grounding"
    "object_identification"
    "object_identification_easy_query"
    "object_identification_with_original"
    "object_identification_image_only"
    "knowledge_extraction"
    "whole"
)

################################
# Vision Language Models
################################
VLLM_MODEL_NAMES=(
    # LLaMA
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # QWEN
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    # GPT (uncomment as needed)
    # "gpt-5-2025-08-07"
    # "gpt-5-mini-2025-08-07"
    # Specialized Models
    # - Visual Grounding
    # "IDEA-Research/grounding-dino-base"
    # "google/owlvit-base-patch32"
)

echo "NUM_SAMPLES: $NUM_SAMPLES"

for TASK_TYPE in "${TASK_TYPE_LIST[@]}"; do
    echo "Running evaluation for TASK_TYPE: $TASK_TYPE"

    DATASET_BASENAME=$(basename "$DATASET")

    if [ "$SPLIT_TYPE" = "full" ]; then
        for MODEL_NAME in "${VLLM_MODEL_NAMES[@]}"; do
            MODEL_BASENAME=$(basename "$MODEL_NAME")

            python evaluate.py \
                --model-name "$MODEL_NAME" \
                --source-dataset-path "$DATASET" \
                --task-type "$TASK_TYPE" \
                --split "$SPLIT" \
                --num-conversations "$NUM_SAMPLES" \
                --output-dir "./output/$DATASET_BASENAME/$TASK_TYPE/$MODEL_BASENAME"

            echo "<$SPLIT_TYPE> Evaluating model [$MODEL_NAME] on dataset [$DATASET] with task type [$TASK_TYPE]"
        done
    else
        for MODEL_NAME in "${VLLM_MODEL_NAMES[@]}"; do
            MODEL_BASENAME=$(basename "$MODEL_NAME")

            python evaluate.py \
                --model-name "$MODEL_NAME" \
                --source-dataset-path "$DATASET" \
                --task-type "$TASK_TYPE" \
                --split "$SPLIT" \
                --num-conversations "$NUM_SAMPLES" \
                --use-ambiguous-label-only \
                --output-dir "./output/$DATASET_BASENAME/$TASK_TYPE/DISAMBG/$MODEL_BASENAME"

            echo "<$SPLIT_TYPE> Evaluating model [$MODEL_NAME] on dataset [$DATASET] with task type [$TASK_TYPE]"
        done
    fi
done
