# Setup

```bash
$ conda create -n cragmm_diagnostics python=3.10
$ conda activate cragmm_diagnostics
$ pip install -r requirements.txt
```

# Evaluation

## Decoupled Stage Evaluation

```bash
bash ./batch_eval.sh
```

### 1. Language-based Visual Grounding
- `visual_grounding` : Given image and original query, the task is to find target region's coordinates as output.
 
### 2. Object Identification

We separate input variants to enable understanding of each modality for object identification task.

#### Task Types
- `object_identification`: Target region highlighting with bounding box and original query.
- `object_identification_easy_query`: Target region highlighting with bounding box; uses identification prompt "what is the name of the object" as query.
- `object_identification_with_original`: Original image and query are used together.
- `object_identification_image_only`: Uses only original image with identification prompt "what is the name of the object" as query.

### 3. Knowledge-intensive Question Answering

#### Task Types

- `knowledge_extraction`: Uses textual-only questions. (e.g., Image + When is the production year of this car? -> **When is the production year of Toyota Prius?**)
- `whole`: Uses original image and text together to solve QA tasks.


##### With Retrieval Augmentation (RAG)

**Prerequisites:**
- Generate indexing files for each retriever 

```bash
bash ./eval_rag.sh
```

