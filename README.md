# PathSnap — GPS Trajectory Map Matching

PathSnap is a GPS trajectory map-matching project designed to reconstruct road-network routes from noisy GPS observations.

The project combines data preprocessing, feature engineering, machine-learning models, route reconstruction, and a web-based interface.

## Project Objectives

- Process GPS trajectory data.
- Extract useful features for road matching.
- Compare machine-learning approaches.
- Explore Bayesian hyperparameter optimization.
- Use Hidden Markov Models (HMM) and Viterbi decoding for route reconstruction.
- Provide an API backend and an interactive map interface.

## Project Architecture

The project has two main components:

**1. Backend — FastAPI**
- Loads project components and model-related resources.
- Provides health and trip-list endpoints.
- Defines the prediction endpoint.

**2. Frontend — Streamlit**
- Provides the user interface.
- Displays map-related information using Folium.
- Communicates with the backend API.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Data processing | NumPy, Pandas, SciPy |
| Machine learning | Scikit-learn, LightGBM |
| Optimization | Optuna |
| Geospatial processing | PyProj, Shapely |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit, Folium |
| Testing | Pytest |

## Repository Structure

```text
PATH-SNAP/
│
├── .github/          # GitHub configuration
├── api/              # FastAPI backend
├── artifacts/        # Saved model artifacts
├── configs/          # Project configuration
├── data/             # Data-related files
├── docs/             # Project documentation
├── instructions/     # Project instructions
├── notebooks/        # Jupyter notebooks
├── pathsnap/         # Core project modules
├── report/           # Technical report
├── results/          # Experimental results
├── scripts/          # Execution and utility scripts
├── tests/            # Automated tests
├── ui/               # Streamlit frontend
│
├── README.md
├── requirements.txt
└── pytest.ini
```

## Dataset Setup

The dataset is not included in this repository.

Obtain the dataset separately from its authorized source and follow the expected directory structure and configuration described in the project documentation.

Do not upload private, restricted, or evaluator-only datasets to GitHub.

The project configuration uses dataset directories for training and evaluation. Check `configs/default.yaml` before running data-dependent scripts.

## Installation

### Prerequisites

- Python 3.11 recommended.
- Git.
- VS Code or another Python development environment.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PATH-SNAP.git
cd PATH-SNAP
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install backend and project dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```powershell
pip install -r ui/requirements.txt
```

## Running the Application

### Start the Backend API

From the project root, with the virtual environment activated:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

Check the health endpoint:

http://127.0.0.1:8000/health

### Start the Streamlit Frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
streamlit run ui/app.py
```

Streamlit will display the local address for the frontend in the terminal.

Make sure the backend is running before using frontend features that require API communication.

## Running Tests

From the project root:

```powershell
pytest -v
```

Some data-dependent tests require the dataset to be available locally.

## Current Implementation Status

The repository contains preprocessing, feature engineering, model-related components, route reconstruction code, API routes, and a Streamlit interface.

**Important API limitation:** The `/predict` endpoint is currently not implemented and returns HTTP 501. Therefore, the complete prediction workflow should not be considered operational yet.

Some tests also require dataset files that are not included in this repository.

## Important Notes

- Keep datasets and private credentials outside public Git repositories.
- Do not commit virtual environments or generated cache files.
- Use the pinned dependency versions in `requirements.txt` to reduce environment compatibility issues.
- Review the project documentation before running individual training or evaluation scripts.

## Acknowledgements

This project is based on the PathSnap project and its associated project materials. Please retain the original authorship information and obtain the necessary permission before redistributing the original source code.

