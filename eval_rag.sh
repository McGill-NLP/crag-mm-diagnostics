#!/bin/bash
################################
# Configuration Variables
################################
NUM_SAMPLES=-1  # -1 for all samples
DATASET=McGill-NLP/crag-mm-diagnostic
SPLIT_TYPE=full  # Options: full, disambiguous_only
SPLIT=test

################################
# Model Configuration
################################
MODEL_TYPE=AdvancedRAG  # Options: AdvancedRAG, PrecomputedRetrievalAdvancedRAG, Retriever
PREBUILT_RETRIEVAL_INFO=./data/id2_retrieval_info.pickle  # Only used for PrecomputedRetrievalAdvancedRAG
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="gpt-5-2025-08-07"

################################
# Retriever Configuration
################################
RETRIEVER_TYPE="separate"  # Options: "mllm", "separate"
RETRIEVER_MODEL="clip"  # "Qwen/Qwen3-VL-Embedding-2B"
WEB_HF_INDEX_NAME="crag-mm-2025/web-search-index-public-test"
IMAGE_HF_INDEX_NAME="crag-mm-2025/image-search-index-public-test"

################################
# RAG Type Configuration
################################
# Used when MODEL_TYPE="PrecomputedRetrievalAdvancedRAG"
RAG_TYPE="both_dino_cropped_image_text"
# Available options:
# - "image_only", "dino_cropped_image_only", "gt_cropped_image_only"
# - "text_only", "text_only_w_textual_only_query"
# - "both_normal_image_text", "both_dino_cropped_image_text", "both_gt_cropped_image_text"
# - "no_augmentation", "no_augmentation_text_only"

################################
# Derived Variables
################################
MODEL_BASENAME=$(basename "$MODEL_NAME")
DATASET_BASENAME=$(basename "$DATASET")
RETRIEVER_BASENAME=$(basename "$RETRIEVER_MODEL")

echo "NUM_SAMPLES: $NUM_SAMPLES"

if [ "$MODEL_TYPE" = "RetrievalSanityChecker" ]; then
    OUTPUT_DIR=./output/$DATASET_BASENAME/Retriever/${MODEL_TYPE}_${RETRIEVER_TYPE}_${RETRIEVER_BASENAME}_${RAG_TYPE}_${NUM_SAMPLES}
    echo "Running [$MODEL_TYPE] on dataset [$DATASET] with Retrieval type [$RAG_TYPE]"

    python evaluate_rag.py \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --num-conversations "$NUM_SAMPLES" \
        --display-conversations 3 \
        --model-type "$MODEL_TYPE" \
        --model-name "$MODEL_NAME" \
        --retriever-type "$RETRIEVER_TYPE" \
        --retriever-model-name "$RETRIEVER_MODEL" \
        --web-hf-index-id "$WEB_HF_INDEX_NAME" \
        --image-hf-index-id "$IMAGE_HF_INDEX_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --eval-model None

elif [ "$MODEL_TYPE" = "AdvancedRAG" ]; then
    if [ "$SPLIT_TYPE" = "full" ]; then
        OUTPUT_DIR=./output/$DATASET_BASENAME/whole/${MODEL_TYPE}_${MODEL_BASENAME}_${RETRIEVER_TYPE}_${RETRIEVER_BASENAME}_${RAG_TYPE}_${NUM_SAMPLES}
        echo "Running [$MODEL_TYPE] on dataset [$DATASET]"
        python evaluate_rag.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-conversations "$NUM_SAMPLES" \
            --display-conversations 3 \
            --model-type "$MODEL_TYPE" \
            --model-name "$MODEL_NAME" \
            --retriever-type "$RETRIEVER_TYPE" \
            --retriever-model-name "$RETRIEVER_MODEL" \
            --web-hf-index-id "$WEB_HF_INDEX_NAME" \
            --image-hf-index-id "$IMAGE_HF_INDEX_NAME" \
            --output-dir "$OUTPUT_DIR"

    else
        OUTPUT_DIR=./output/$DATASET_BASENAME/whole/DISAMBG/${MODEL_TYPE}_${MODEL_BASENAME}_${RETRIEVER_TYPE}_${RETRIEVER_BASENAME}_${RAG_TYPE}_${NUM_SAMPLES}
        echo "Running [$MODEL_TYPE] on dataset [$DATASET]"
        python evaluate_rag.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-conversations "$NUM_SAMPLES" \
            --display-conversations 3 \
            --model-type "$MODEL_TYPE" \
            --model-name "$MODEL_NAME" \
            --retriever-type "$RETRIEVER_TYPE" \
            --retriever-model-name "$RETRIEVER_MODEL" \
            --web-hf-index-id "$WEB_HF_INDEX_NAME" \
            --image-hf-index-id "$IMAGE_HF_INDEX_NAME" \
            --output-dir "$OUTPUT_DIR" \
            --use-ambiguous-label-only
    fi
else
    if [ "$SPLIT_TYPE" = "full" ]; then
        OUTPUT_DIR=./output/$DATASET_BASENAME/whole/${MODEL_TYPE}_${MODEL_BASENAME}_${RETRIEVER_TYPE}_${RETRIEVER_BASENAME}_${RAG_TYPE}_${NUM_SAMPLES}
        echo "Running [$MODEL_TYPE] on dataset [$DATASET] with Retrieval type [$RAG_TYPE]"
        python evaluate_rag.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-conversations "$NUM_SAMPLES" \
            --display-conversations 3 \
            --model-type "$MODEL_TYPE" \
            --prebuilt-retrieval-info-path "$PREBUILT_RETRIEVAL_INFO" \
            --model-name "$MODEL_NAME" \
            --retriever-type "$RETRIEVER_TYPE" \
            --output-dir "$OUTPUT_DIR" \
            --retriever-model-name "$RETRIEVER_MODEL" \
            --rag-type "$RAG_TYPE"
    else
        OUTPUT_DIR=./output/$DATASET_BASENAME/whole/DISAMBG/${MODEL_TYPE}_${MODEL_BASENAME}_${RETRIEVER_TYPE}_${RETRIEVER_BASENAME}_${RAG_TYPE}_${NUM_SAMPLES}
        echo "Running [$MODEL_TYPE] on dataset [$DATASET] with Retrieval type [$RAG_TYPE]"
        python evaluate_rag.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --num-conversations "$NUM_SAMPLES" \
            --display-conversations 3 \
            --model-type "$MODEL_TYPE" \
            --prebuilt-retrieval-info-path "$PREBUILT_RETRIEVAL_INFO" \
            --model-name "$MODEL_NAME" \
            --retriever-type "$RETRIEVER_TYPE" \
            --output-dir "$OUTPUT_DIR" \
            --retriever-model-name "$RETRIEVER_MODEL" \
            --use-ambiguous-label-only \
            --rag-type "$RAG_TYPE"
    fi
fi

echo "OUTPUT_DIR: $OUTPUT_DIR"