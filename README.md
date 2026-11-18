# ASP-MT — standalone local installation

This folder contains the pipeline, its data resources, local embedding checkpoint,
and its own Python environment. It does not need the original development project.
The approved flat Python source layout and threshold-T acceptance policy are unchanged.

## Contents

```text
ASP-MT/
  *.py                       Pipeline and the two evaluation entry points
  config.local.json          Paths relative to this folder, not the terminal location
  .env                       Private local UMLS/service settings (never upload)
  .venv/                     This installation's Python dependencies
  resources/
    meddra/                  CSV/raw terminology, vector config, optional UMLS synonyms
      meddra_vector_db/      Existing Qdrant database; 79,026 LLT vectors
    lexicon/                 Four source-derived lexical resources
  models/BioLORD-2023/        Pinned local embedding model and tokenizer
  data/
    training/                terms.csv and cases.csv — 2,927 selected terms
    test/                    terms.csv and cases.csv — 2,715 selected terms
```

The evaluation CSVs are the bound prepared TAC manifests, not the original raw TAC
XML archive. Gold labels enter scoring only, never runtime candidate selection.
The lexical resource location does not make benchmark labels runtime evidence.

## Requirements and checks

Python 3.13 remains an installed system prerequisite. The supplied local `.venv`
uses that interpreter, with its dependencies inside this folder. Ollama is a
separately installed service with the pinned `qwen3:8b` model; it is not copied
into the project. UMLS retrieval requires network access and a valid local key.
Neither Ollama nor UMLS depends on the original project folder.

```powershell
Set-Location 'D:\Agentic Standardization Pipline for MedDra Terms (ASP-MT)'
.\.venv\Scripts\python.exe -B .\preflight.py
```

Preflight is offline: it checks local paths, versions, reference/model hashes and
that this installation's Python environment is being used. It does not call
Ollama or UMLS. Run one process per local vector database and output directory.

## Run

For an input JSON list such as
`[{"id":"INPUT_001","text":"headache","frequency":1}]`:

```powershell
.\.venv\Scripts\python.exe .\run_pipeline.py --input .\inputs.json --directory .\runs\terms_1 --env-file .\.env
```

These evaluation commands invoke the model/retrieval services, then automatically
score and print the concise summary:

```powershell
# Ten-term check; choose a fresh run directory.
.\.venv\Scripts\python.exe .\evaluate_training.py run --limit 10 --directory .\runs\training10_1 --env-file .\.env

# Full training and test runs, separately.
.\.venv\Scripts\python.exe .\evaluate_training.py run --directory .\runs\training_1 --env-file .\.env
.\.venv\Scripts\python.exe .\evaluate_test.py run --directory .\runs\test_1 --env-file .\.env
```

Use `--resume` with the same run arguments after an interruption. Completed and
failed attempts are preserved, not automatically retried. The fixed model-call
ceiling is 12 times the number of inputs; validation attempt limits remain three.
Scores and audits are written within the run folder. Offline reprinting/scoring:

```powershell
.\.venv\Scripts\python.exe .\evaluate_training.py summary --score .\runs\training10_1\evaluation.json
.\.venv\Scripts\python.exe .\evaluate_test.py score --directory .\runs\test_1 --output .\outputs\test_rescore_1.json
```

## Summary meaning

The table reports **PT exact gold**, **LLT exact gold**, and **LLT supported
agreement**, each with coverage, accuracy/agreement and disagreement counts.
Supported agreement credits exact or independently verified lexical matches when
LLT gold exists; a verified current LLT-parent match to gold PT is permitted only
when LLT gold is genuinely absent. It does not turn every same-PT LLT into a
correct labelled LLT. Complete-set and independent clinical checks are unchanged.

Coverage uses eligible terms. Accuracy/agreement uses automatic outputs. Outcome
percentages use all selected terms: automatic PT+LLT matches, human review, or
explicit no-supported-match outcomes. Technical errors and unexecuted inputs are
separate, and remain in eligibility denominators. Detailed metrics and overlap
diagnostics stay in JSON. The prior test has 1,110 training-wording overlaps out
of 2,715; combined results are not an unseen-wording result.

## Moving or publishing

Resource paths are relative to `config.local.json`, independent of the working
directory. On another machine, recreate `.venv` with Python 3.13 and install
`requirements.txt`; a virtual environment itself is not a portable Python
distribution. Install/configure the required Ollama model separately. Existing
run caches bind code/resource paths and must not be silently reused after moving.

This local folder includes private/licensed resources for your own installation.
**Do not upload the whole folder as a ZIP to GitHub.** `.gitignore` excludes
`.env`, private config, `.venv`, resources, models, data and outputs. Publish the
source and setup instructions only unless you have redistribution permission.
Other users must supply the required resources and their own credentials.
Do not run `configure.py` here again: the relative configuration already exists.
For a fresh installation, it writes that configuration once local resources are
placed at the listed paths and refuses to overwrite an existing file.

No new clinical performance result is claimed by this packaging change.
