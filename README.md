# TabEmb

Code and sample data for the ACL 2026 paper **"TabEmb: Joint Semantic-Structure Embedding for Table Annotation"**.

TabEmb combines:

- LLM-based column embeddings for table content.
- A graph neural network (GNN) that injects table structure.
- Support for table annotation tasks including column type annotation (CTA), column property annotation (CPA), and table type annotation (TTA).

## Repository Contents

- `src/Save_embedding.py`: generates column embeddings from table columns with an LLM.
- `src/GNN.py`: trains the downstream GNN on saved embeddings.
- `src/webtables_GNN_5folds.py`: 5-fold WebTables baseline runner.
- `data/graph_labels/`: graph-level labels required for the released T2D setup.
- `data/representation/dataset-t2d/`: sample precomputed T2D embeddings included in this release.
- `src/labels/`: label vocabularies used by the released experiments.

## Release Scope

This repository currently includes the codebase plus released artifacts for the T2D setup. The bundled sample representations are sufficient to run the GNN stage for `dataset-t2d` without regenerating embeddings.

The embedding-generation code is included, but running it for datasets beyond the bundled sample data requires you to prepare the corresponding input CSV files locally.

## Environment

- Python 3.10 is recommended.
- A GPU is strongly recommended for embedding generation.
- Installing DGL and PyTorch should match your CUDA setup.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use gated Hugging Face models, export a token first:

```bash
export HF_TOKEN=your_token_here
```

## Data Format

`src/Save_embedding.py` expects CSV files with at least these columns:

- `data`: serialized column content, with row values separated by ` [SEP] `
- `class_id`: integer label
- `table_id`: table identifier
- `col_idx`: column index

Dataset-specific CSV filenames are currently configured inside `src/Save_embedding.py`.

## Quick Start

### 1. Train the released T2D GNN setup

Use the included precomputed embeddings:

```bash
python src/GNN.py --task cta --data-name dataset-t2d
```

For CPA:

```bash
python src/GNN.py --task cpa --data-name dataset-t2d
```

### 2. Generate embeddings

Example:

```bash
python src/Save_embedding.py --data-name dataset-t2d --model-name mistral
```

Embeddings are written to:

```text
data/representation/<data-name>/<task>/
```

The current release writes CTA-format embeddings using the naming convention expected by `src/GNN.py`.

### 3. Run the WebTables 5-fold baseline

```bash
python src/webtables_GNN_5folds.py --task cta --data-name webtables
```

## Model Names

Supported `--model-name` values are defined in `src/model_llm.py`, including:

- `mistral`
- `mistral-instruct`
- `mixtral`
- `mixtral-instruct`
- `llama2`
- `llama3`
- `llama3-instruct`
- `llama3.1`
- `llama3.1-instruct`
- `phi3`
- `Qwen7`
- `Qwen1.5`
- `Qwen14`

## Dataset Sources

- SOTAB: <https://webdatacommons.org/structureddata/sotab/v2/>
- WikiTable / TURL: <https://github.com/sunlab-osu/TURL>
- WebTables / SATO: <https://github.com/megagonlabs/sato>
- T2D-related resources:
  - <https://github.com/wbsg-uni-mannheim/TabAnnGPT/tree/main/CPAusingLLMs>
  - <https://github.com/alan-turing-institute/SemAIDA>

## Notes

- `src/GNN.py` is now CLI-driven and no longer requires interactive dataset selection.
- The released workflow is best-supported for the included `dataset-t2d` artifacts.
- License metadata is not included in this repository yet. Add the project license before making the repository public.
