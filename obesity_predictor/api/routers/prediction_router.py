"""
prediction_router.py
=========================
Prediction endpoint for obesity classification.

Features
--------
- Validates incoming requests with Pydantic
- Preprocesses data before prediction
- Uses InferencePipeline for end-to-end inference
- Logs predictions and request metadata

Author: Rostand Surel
"""

from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
#from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings
from obesity_predictor.core.validation.schema_validator import validate_input_records
from obesity_predictor.core.pipeline.inference_pipeline import InferencePipeline

router = APIRouter()


@router.post("/")
def predict(records: List[dict]):
    """
    Run obesity prediction for one or more input records.

    Example Request
    ---------------
    ```json
    [
      {
        "Gender": "Male",
        "Age": 25,
        "Height": 175,
        "Weight": 70,
        "family_history_with_overweight": "Yes",
        "FAVC": "Yes",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "No",
        "CH2O": 2.0,
        "SCC": "No",
        "FAF": 2.0,
        "TUE": 1.0,
        "CALC": "Sometimes",
        "MTRANS": "Public_Transport"
      }
    ]
    ```

    Returns
    -------
    dict
        Predictions for each record.
    """
    try:
        #logger.info(f"[API] Received {len(records)} record(s) for prediction.")
        validated = validate_input_records(records)
        df = pd.DataFrame([v.dict() for v in validated])

        # Inference pipeline
        model_path = f"{settings.artifact_dir}/{settings.best_model_name}_model.joblib"
        preproc_path = f"{settings.artifact_dir}/{settings.best_model_name}_preprocessor.joblib"

        pipeline = InferencePipeline(model_path=model_path, preprocessor_path=preproc_path)
        result = pipeline.predict(df)

        #logger.success("[API] Prediction completed successfully.")
        return result

    except Exception as e:
        #logger.error(f"[API] Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))