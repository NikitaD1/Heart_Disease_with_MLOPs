"""Request/response schemas for the prediction API (validated by Pydantic)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


class PatientFeatures(BaseModel):
    """The 13 clinical attributes of the UCI Cleveland dataset.

    `ca` and `thal` may be null (they are missing in the source data too); the
    model pipeline imputes them with the training-set statistics.
    """
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="forbid")

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain: 1 typical, 2 atypical, "
                                                     "3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels coloured (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = heart disease, 0 = no heart disease")
    label: str
    probability_disease: float = Field(..., description="Model probability of disease")
    confidence: float = Field(..., description="Probability of the predicted class")
    model_version: str


class BatchRequest(BaseModel):
    instances: list[PatientFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
