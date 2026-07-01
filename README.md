# Deep Learning Submission Pack

This folder is a cleaned submission copy of the current UniForm project. It keeps the runtime application stack in one place, while separating training, preprocessing, evaluation, and model artifacts so another person can reproduce the project on a new machine with less hunting around.

## Folder layout

- `app/backend/`
  - FastAPI services for ODC, KVP, MinIO storage, auth, and gold-tier transformation.
  - Runtime backend models live in `app/backend/models/`.
- `app/geolayoutlm/`
  - GeoLayoutLM KVP service, upstream GeoLayoutLM code, FUNSD-style preprocessing files, and finetuned checkpoints.
- `app/frontend/odc-next-ui/`
  - Next.js demo UI.
- `app/infra/`
  - Docker Compose for Postgres + MinIO + Dremio, env example, and SQL schema.
- `app/scripts/`
  - Start, stop, and DB bootstrap helpers.
- `app/legacy-streamlit/`
  - Streamlit demo snapshot.
- `train/uniform/`
  - Training scripts and notebooks from the current project.
- `preprocess/uniform/`
  - Dataset preparation, synthetic data generation, OCR cache, and packaging scripts.
- `evaluation/uniform/`
  - Benchmark and evaluation scripts.
- `utilities/root-tools/`
  - Miscellaneous helper scripts copied from the current project root.
- `models/training/`
  - Training-only model artifacts that are not needed by the main app runtime.

## What is included

- Current UniForm runtime services from `SelfCodeDL`
- Next.js demo UI
- GeoLayoutLM service code and checkpoint-loading structure
- Training, preprocessing, and evaluation scripts
- A cleaned infrastructure setup for local replication

## What is intentionally not included

- Generated logs
- Running process artifacts
- `node_modules`, `.next`, `.git`, and cache folders
- Large generated datasets and upload/output folders from the live workspace
- Large model weights and checkpoints that make Git pushes impractical

## Model files to place manually

This repository is meant to be pushed without the large weights. After cloning on a new machine, place the model files manually in the following locations:

- ODC runtime model
  - Put `odc_projection_scripted.pt` in `app/backend/models/`
- Alternate training/runtime projection model
  - Put `full_projection_model.pt` in `models/training/`
- Alternate LayoutLMv3 KVP model snapshot
  - Put the LayoutLMv3 model directory contents in `app/backend/models/Key-Value-Pair/`
  - Expected files include:
    - `config.json`
    - `model.safetensors`
    - `processor_config.json`
    - `tokenizer.json`
    - `tokenizer_config.json`
- GeoLayoutLM pretrained base weight
  - Put `geolayoutlm_large_pretrain.pt` in `app/geolayoutlm/GeoLayoutLM/`
- GeoLayoutLM finetuned checkpoints
  - Put the `.ckpt` checkpoint files in `app/geolayoutlm/geolayoutlm_workspace/checkpoints/`
  - Example filenames:
    - `epoch=40-f1_linking=0.9645.ckpt`
    - `epoch=44-f1_labeling=0.9412.ckpt`

If those files are missing, the code structure will still be present, but the corresponding services will not start successfully.

## Machine requirements

- Windows with PowerShell
- Python 3.10 or 3.11 recommended
- Node.js 20+
- Docker Desktop
- Tesseract OCR installed locally
- Google Cloud CLI installed and authenticated if you want Document AI OCR
- Access to the target Google Document AI processor endpoint

## Python setup

From this folder:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Frontend setup

```powershell
cd .\\app\\frontend\\odc-next-ui
Copy-Item .env.local.example .env.local
npm install
cd ..\\..\\..
```

Edit `app/frontend/odc-next-ui/.env.local` if you want different service URLs or console URLs.

## Infrastructure setup

The cleaned local stack is in `app/infra/docker-compose.yml`. It starts:

- Postgres with pgvector on `5432`
- MinIO API on `9000`
- MinIO Console on `9001`
- Dremio on `9047`

To start only infrastructure:

```powershell
docker compose -f .\\app\\infra\\docker-compose.yml up -d
python .\\app\\scripts\\init_backend_db.py
```

## Start the full application

```powershell
powershell -ExecutionPolicy Bypass -File .\\app\\scripts\\start_submission_stack.ps1 -RebuildNext
```

This starts:

- Next UI: `http://127.0.0.1:8001`
- ODC service: `http://127.0.0.1:8005`
- GeoLayoutLM KVP service: `http://127.0.0.1:8006`
- MinIO storage service: `http://127.0.0.1:8007`
- Auth service: `http://127.0.0.1:8008`
- Gold-tier service: `http://127.0.0.1:8009`

To stop everything:

```powershell
powershell -ExecutionPolicy Bypass -File .\\app\\scripts\\stop_submission_stack.ps1
```

## Environment variables

Use `app/infra/.env.example` as the reference list. The most important ones are:

- `REALFORM_DSN`
- `DOCUMENT_AI_ENDPOINT`
- `GCLOUD_CMD`
- `TESSERACT_CMD`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `ODC_BACKEND_URL`
- `KV_BACKEND_URL`
- `STORAGE_BACKEND_URL`
- `AUTH_BACKEND_URL`
- `GOLD_BACKEND_URL`

## Notes on OCR and replication

- `app/backend/ODCService.py` has been adjusted in this submission copy so model path, Tesseract path, Google Cloud CLI path, Document AI endpoint, and Postgres DSN can all be overridden with environment variables.
- `app/backend/KVExtractService.py` has been adjusted so its model directory resolves cleanly inside this pack.
- The main KVP service for the running app is the GeoLayoutLM service in `app/geolayoutlm/GeoLayoutLMKVExtractService.py`.
- The plain LayoutLMv3 KVP service in `app/backend/KVExtractService.py` is still included as an alternate service snapshot.
- If you keep models somewhere else on disk, you can override the defaults with environment variables such as `ODC_MODEL_PATH`, `KVP_MODEL_DIR`, and `GEOLAYOUTLM_CHECKPOINT`.

## Reference files

- Clean Docker stack: `app/infra/docker-compose.yml`
- Original source Docker stack snapshot: `app/infra/docker-compose.source.yml`
- Clean template schema: `app/infra/sql/01_create_templates.sql`
- Original source SQL snapshot: `app/infra/sql/pgdb.source.sql`

## Suggested first replication flow

1. Create Python env and install `requirements.txt`
2. Install Node dependencies in `app/frontend/odc-next-ui`
3. Manually place the required model files into the paths listed above
4. Copy `app/frontend/odc-next-ui/.env.local.example` to `.env.local`
5. Update `DOCUMENT_AI_ENDPOINT` and Google Cloud auth on the machine
6. Start the stack with `start_submission_stack.ps1`
7. Log in to the UI and test upload, template match, extraction, MinIO storage, and gold-tier transformation
