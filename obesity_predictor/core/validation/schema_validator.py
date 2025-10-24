"""
schema_validator.py
=========================
Defines Pydantic models to validate input data for training
and inference pipelines.

Ensures that data received in production (e.g., API requests)
match the expected schema used during training.

Author: Rostand Surel
"""

from pydantic import BaseModel, Field, ValidationError, validator
from typing import Optional


class ObesityInputSchema(BaseModel):
    """
    Schema representing a single data record for inference.
    """

    Gender: str = Field(..., description="Gender of the person ('Male'/'Female')")
    Age: float = Field(..., ge=0, le=120)
    Height: float = Field(..., gt=0)
    Weight: float = Field(..., gt=0)
    family_history_with_overweight: str
    FAVC: str = Field(..., description="Frequent consumption of high-caloric food (Yes/No)")
    FCVC: float = Field(..., ge=0, le=3, description="Frequency of consumption of vegetables")
    NCP: float = Field(..., ge=0, le=5, description="Number of main meals per day")
    CAEC: str = Field(..., description="Consumption of food between meals (Yes/No/Sometimes)")
    SMOKE: str = Field(..., description="Smoking habit (Yes/No)")
    CH2O: float = Field(..., ge=0, le=5, description="Daily water intake (liters)")
    SCC: str = Field(..., description="Calorie consumption monitoring (Yes/No)")
    FAF: float = Field(..., ge=0, le=5, description="Physical activity frequency per week")
    TUE: float = Field(..., ge=0, le=3, description="Time using technology devices per day")
    CALC: str = Field(..., description="Alcohol consumption frequency (No/Sometimes/Frequently)")
    MTRANS: str = Field(..., description="Transportation method (Walking/Bike/Car/Public_Transport)")

    @validator("Gender")
    def gender_must_be_valid(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @validator("family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS")
    def validate_categorical_fields(cls, v):
        if not isinstance(v, str):
            raise ValueError("Expected a string for categorical fields")
        return v


def validate_input_records(records: list[dict]) -> list[ObesityInputSchema]:
    """
    Validate a list of input records for inference.

    Parameters
    ----------
    records : list[dict]
        List of JSON-like records received via API.

    Returns
    -------
    list[ObesityInputSchema]
        List of validated, strongly typed Pydantic models.

    Raises
    ------
    ValidationError
        If any record fails schema validation.
    """
    validated = []
    for record in records:
        try:
            validated.append(ObesityInputSchema(**record))
        except ValidationError as e:
            raise ValueError(f"Invalid record format: {e}")
    return validated