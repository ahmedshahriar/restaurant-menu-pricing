# MLflow Model Image

This folder contains the **Dockerfile** and related artifacts used to build and serve the MLflow model.

---

## 📦 Contents
- `Dockerfile` — generated via `mlflow models generate-dockerfile`
- `model_dir/` — auto-downloaded MLflow model artifacts (not committed)
- `.dockerignore` — restricts build context for smaller, faster builds
- `README.md` — usage and maintenance notes

---

## 🏗️ Build Instructions

Run from the root of the repository:

```bash
# Ensure the MLflow tracking server is accessible (modify the port if needed)
export MLFLOW_TRACKING_URI=http://localhost:8080

# Generate the Dockerfile + model_dir (if not yet done)
mlflow models generate-dockerfile \
  --model-uri "models:/<your_model_name>/latest" \
  --output-directory ./infrastructure/docker/mlflow-models/menu-price

# Change to the Dockerfile directory
cd infrastructure/docker/mlflow-models/menu-price

# Build the Docker image
docker build -t menu-price-api .

# Run the Docker container (modify the port if needed)
docker run -p 5001:8080 --name menu-price-api-container menu-price-api
```
