# AgroVision

Fruit freshness classification API built with FastAPI and PyTorch. The model classifies images of fruits (apple, banana, bitter gourd, capsicum, orange, tomato) as either fresh or stale.

## Project Structure

```
agrovision/
├── artifacts/                  # Model artifacts loaded at startup
│   ├── classes.json            # Class label definitions
│   ├── class_weights.json      # Class weights used during training
│   └── normalization_stats.json # Image normalization mean/std
│
├── data/
│   └── raw/                    # Raw image dataset organized by class
│
├── notebooks/                  # Jupyter notebooks for the ML pipeline
│   ├── 00_data_acquisition.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                        # Application source code
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Settings loaded from environment / .env
│   └── features/
│       └── freshness/          # Freshness classification feature
│           ├── freshness_controller.py  # Route definitions
│           ├── freshness_service.py     # Model inference logic
│           ├── dtos/                    # Response data transfer objects
│           └── pipes/                   # Request processing (image decoding)
│
├── tests/                      # Pytest test suite
│   ├── conftest.py
│   ├── test_main.py
│   └── features/
│       └── freshness/
│
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── .env                        # Environment variables (not committed)
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For development (includes testing tools):

```bash
pip install -r requirements-dev.txt
```

### 3. Configure environment variables

Create a `.env` file at the project root:

```env
MLFLOW_SERVER_URI=http://localhost:5000
MLFLOW_MODEL_URI=models:/<your-model-name>/<version-or-alias>
ALLOWED_ORIGINS=["http://localhost:3000"]
```

| Variable | Description |
|---|---|
| `MLFLOW_SERVER_URI` | URI of the running MLflow tracking server |
| `MLFLOW_MODEL_URI` | MLflow model URI used to load the registered model |
| `ALLOWED_ORIGINS` | JSON array of allowed CORS origins |

### 4. Pull model artifacts

Make sure the `artifacts/` directory contains `classes.json` and `normalization_stats.json` before starting the server. These can be retrieved via DVC:

```bash
dvc pull
```

## Running the API

```bash
fastapi run src/main.py
```

The server starts on `http://0.0.0.0:8000` by default.

For development with auto-reload:

```bash
fastapi dev src/main.py
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/freshness/classify` | Classify a fruit image as fresh or stale |

Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

### Example request

```bash
curl -X POST http://localhost:8000/freshness/classify \
  -F "file=@/path/to/fruit_image.jpg"
```

## Running Tests

```bash
pytest
```
