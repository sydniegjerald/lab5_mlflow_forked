# app/server.py
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)

# ---- Runtime-served version state (minimal additions) ----
CURRENT_SERVED_VERSION = MODEL_VERSION  # track which version is currently served

def _load_model_for_version(version: str):
    """Load a model version from MLflow Model Registry."""
    uri = f"models:/{MODEL_NAME}/{version}"
    return mlflow.pyfunc.load_model(uri)


# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    # TODO Run predict
    return PredictResponse(
        class_id=[],
        class_label=[]
    )
    
# TODO Add endpoint to get the current model serving version
# TODO Add endpoint to update the serving version
# TODO Predict using the correct served version

from pydantic import BaseModel

class SelectVersionRequest(BaseModel):
    version: str

@app.get("/model/version")
def get_model_version():
    """Return the currently served model version."""
    return {"model_name": MODEL_NAME, "version": CURRENT_SERVED_VERSION}

@app.post("/model/version")
def set_model_version(req: SelectVersionRequest):
    """Switch the served model to a specific version (registered in MLflow)."""
    global model, CURRENT_SERVED_VERSION
    try:
        loaded = _load_model_for_version(req.version)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load version '{req.version}': {e}")
    model = loaded
    CURRENT_SERVED_VERSION = req.version
    return {"model_name": MODEL_NAME, "version": CURRENT_SERVED_VERSION}

