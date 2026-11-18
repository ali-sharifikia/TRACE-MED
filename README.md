# TRACE-MED

TRACE standardizes clinical expressions to MedDRA Preferred Terms (PTs) and Lowest Level Terms (LLTs). It combines terminology retrieval, LLM-based interpretation and review, deterministic validation, and rule-based completion.

The pipeline supports custom inputs, training and test evaluation, retrieval evaluation, and optional per-input decision recording and visualization.

## How it works

1. **Normalization and exact matching:** Normalize the input and apply the existing current-LLT exact-matching rules.
2. **Input interpretation:** Identify clinical concepts, components, negation, and unresolved ambiguity.
3. **Candidate retrieval:** Retrieve candidates through fuzzy matching, BioLORD semantic retrieval, and UMLS lookup.
4. **Candidate aggregation:** Consolidate candidates by LLT code while preserving official identities, parent PTs, source scores, and query provenance.
5. **Reviewer assessment:** Compare PT groups and their LLTs. The reviewer can propose a mapping, request additional evidence, recommend human review, or report no supported match.
6. **Deterministic validation and acceptance:** Check candidate identities, clinical consistency, interpretation scope, and the applicable acceptance requirements.
7. **Completion:** Apply the configured deterministic completion policies to eligible unresolved mappings.
8. **Final output:** Return an automatic mapping, human-review outcome, or no-supported-match outcome. Execution failures are recorded separately.

An unresolved LLT can retain an accepted PT. Shared PT membership alone does not establish LLT equivalence.

## Requirements

- Python 3.13.
- Dependencies listed in `requirements.txt`.
- A running local Ollama service with the `qwen3:8b` model matching `model.lock.json`.
- The local BioLORD-2023 checkpoint matching `embedding.lock.json`.
- The required MedDRA reference, populated vector database, and lexical resources.
- A valid UMLS API key and network access for UMLS retrieval.
- Prepared training and test CSV files for dataset evaluation.

The current PNG visualization backend uses Windows PowerShell and System.Drawing.

Required resource files are checked against the existing resource bindings. The pipeline does not automatically download models or build missing terminology resources.

## Project layout

```text
TRACE/
├── run_pipeline.py                 Custom-input execution
├── evaluate_training.py            Training evaluation
├── evaluate_test.py                Test evaluation
├── retrieval_evaluation.py         Offline retrieval evaluation
├── visualize_decision_trace.py     Offline decision visualization
├── validate_decision_trace.py      Trace validation
├── preflight.py                    Offline installation checks
├── configure.py                    Local resource configuration
├── requirements.txt
├── config.local.json
├── .env
├── resources/
│   ├── meddra/
│   │   ├── meddra_exact_reference.csv
│   │   ├── llt.asc
│   │   ├── pt.asc
│   │   ├── meddra_vector_db_config.json
│   │   └── meddra_vector_db/
│   └── lexicon/
│       ├── words.json
│       ├── phrases.json
│       ├── nominalizations.json
│       └── agreement.json
├── models/
│   └── BioLORD-2023/
├── data/
│   ├── training/
│   │   ├── terms.csv
│   │   └── cases.csv
│   └── test/
│       ├── terms.csv
│       └── cases.csv
├── runs/
└── report/
```

The remaining Python modules implement the workflow, retrieval, review, acceptance, completion, and recording mechanisms.

## Environment setup

Run commands from the project folder in PowerShell.

For an installation without a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For an existing environment, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

All commands below assume this environment is active. Alternatively, replace `python` with `.\.venv\Scripts\python.exe`.

Create `.env` from `.env.example` if it does not already exist, then supply your local settings:

```text
UMLS_API_KEY=your_api_key
OLLAMA_HOST=http://localhost:11434
```

After placing the required resources in the documented locations, create the configuration if it is missing:

```powershell
python .\configure.py
```

Skip this command when `config.local.json` already exists; the script refuses to overwrite it.

Check the installation:

```powershell
python .\preflight.py
```

Preflight checks local configuration, dependencies, resources, and embedding bindings. It makes no model or retrieval calls. Ollama availability and the installed reviewer-model digest are checked during execution.

## Run custom inputs

Create `inputs.json` containing a list of records:

```json
[
  {
    "id": "INPUT_001",
    "text": "headache",
    "frequency": 1
  },
  {
    "id": "INPUT_002",
    "text": "increases in serum transaminases",
    "frequency": 1
  }
]
```

Each record requires a unique `id` and nonempty `text`. `frequency` is optional. Input IDs may contain letters, numbers, underscores, and hyphens.

Run:

```powershell
python .\run_pipeline.py --input .\inputs.json --directory .\runs\custom_1 --env-file .\.env --record-decisions
```

This command invokes the pipeline and saves the results. `--record-decisions` enables additional evidence recording without changing mapping decisions.

## Training and test evaluation

Both entry points execute the same configured pipeline, score the saved outputs, and print an evaluation summary. Running the training evaluation does not train or fine-tune the LLM.

For a ten-input training check:

```powershell
python .\evaluate_training.py run --limit 10 --directory .\runs\training10_2 --env-file .\.env --record-decisions
```

For the full prepared training set:

```powershell
python .\evaluate_training.py run --directory .\runs\training_full_recorded_2 --env-file .\.env --record-decisions
```

For the full prepared test set:

```powershell
python .\evaluate_test.py run --directory .\runs\test_full_recorded_2 --env-file .\.env --record-decisions
```

Run these sequentially. Do not run multiple processes against the same local vector database.

Each new experiment requires a fresh run directory. Reference labels are used during scoring, not supplied to the pipeline for candidate selection.

### Evaluation measures

The terminal summary reports total inputs, automatic matches, human-review outcomes, unsupported outcomes, execution errors, and:

| Measure | Meaning |
|---|---|
| PT exact gold | Predicted PT sets match the eligible reference PT sets exactly. |
| LLT exact gold | Predicted LLT sets match the eligible reference LLT sets exactly. |
| LLT supported agreement | Exact or independently verified lexical agreement when LLT annotations exist; verified current-parent agreement with the reference PT set when LLT annotations are genuinely absent and fallback eligibility requirements hold. |

Coverage measures automatic outputs among eligible inputs. Accuracy or agreement measures correct outputs among eligible automatic outputs.

Supported agreement does not treat every same-parent LLT as correct when an LLT reference is available. Missing labels do not establish exact LLT correctness.

Human review and unsupported outcomes are distinct from execution errors. Errors and unexecuted inputs remain in the applicable eligibility denominators.

Test evaluation also saves training-wording overlap diagnostics in its JSON output; overall test results should not be interpreted as performance exclusively on unseen wording.

### Saved evaluation files

Each evaluation run saves:

- `evaluation.json`: summary metrics.
- `evaluation_audit.json`: per-input scoring details.
- `records/`: saved pipeline states and final outcomes.
- `calls/`: model requests, responses, and exceptions.
- `contract.json`: input, configuration, model, and resource bindings.
- `traces/`: decision events and evidence snapshots when recording is enabled.

Later scoring exports use fresh filenames rather than overwriting earlier results.

To print an existing summary without running the pipeline:

```powershell
python .\evaluate_training.py summary --score .\runs\training_full_recorded_2\evaluation.json
python .\evaluate_test.py summary --score .\runs\test_full_recorded_2\evaluation.json
```

To rescore saved outputs offline:

```powershell
python .\evaluate_training.py score --directory .\runs\training_full_recorded_2 --output .\runs\training_full_recorded_2\evaluation_rescore_1.json
python .\evaluate_test.py score --directory .\runs\test_full_recorded_2 --output .\runs\test_full_recorded_2\evaluation_rescore_1.json
```

These scoring commands make no model or retrieval calls.

## Resume an interrupted run

Repeat the original run command with `--resume`:

```powershell
python .\evaluate_training.py run --directory .\runs\training_full_recorded_2 --env-file .\.env --record-decisions --resume
```

Keep the original input selection and arguments. Resume validates the saved run contract and preserves completed, failed, and previously reserved attempts. It processes only inputs that have not already been reserved.

Changing the bound implementation, model, inputs, or resources prevents resuming the existing run.

## Retrieval evaluation

Retrieval evaluation scores saved execution evidence. It pools the prepared training and test records while retaining partition identities and excludes inputs resolved by the initial exact-matching pathway.

Run after both recorded full evaluations:

```powershell
python .\retrieval_evaluation.py score --training-directory .\runs\training_full_recorded_2 --test-directory .\runs\test_full_recorded_2 --directory .\runs\retrieval_full_recorded_2
```

The evaluation compares initial fuzzy, semantic, UMLS, and combined candidate pools, followed by reviewer and expansion checkpoints. It reports PT and LLT recall, complete-set retrieval, and rank-based metrics where a meaningful recorded ranking exists.

Candidate availability is not automatic mapping coverage. Missing evidence and unavailable rankings are reported explicitly.

Results include `summary.json`, `summary.txt`, `per_input.jsonl`, and supporting provenance records.

## Decision visualization

Runs created with `--record-decisions` support offline, per-input decision charts.

For the custom example above:

```powershell
python .\visualize_decision_trace.py --directory .\runs\custom_1 --term-id INPUT_002 --output .\report\input_002_view_1
```

The default output contains:

- `overview.png`: one continuous decision-flow image.
- `rendering_manifest.json`: source references and validation findings.

Optional arguments:

- `--export-format svg`: SVG output.
- `--export-format both`: PNG and SVG output.
- `--include-details`: additional evidence-detail pages.

Use a fresh visualization output directory. Rendering reads saved artifacts and makes no model or retrieval calls.

The chart distinguishes model proposals, deterministic changes, and final outcomes. An incomplete trace indicates missing or inconsistent recording evidence; it does not by itself mean the mapping failed.

Validate all recorded traces in a run:

```powershell
python .\validate_decision_trace.py --directory .\runs\custom_1
```

## Model configuration and prompts

The reviewer uses local `qwen3:8b` with temperature `0.0`, seed `42`, thinking disabled, and a configured context window of 16,384 tokens.

- Runtime settings: `pipeline.py`
- Interpreter system prompt: `INPUT_INTERPRETATION_PROMPT` in `input_interpretation.py`
- Reviewer system prompt: `REVIEWER_SYSTEM_PROMPT` in `reviewer_prompt.py`
- Reviewer request construction and generation: `reviewer_llm.py`

Exact model requests and responses are retained in each run's `calls/` directory.
