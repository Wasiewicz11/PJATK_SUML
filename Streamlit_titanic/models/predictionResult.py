from pydantic import BaseModel, Field

class PredictionResult(BaseModel):
    survived: bool = Field(..., description="Czy pasażer przeżył (True/False)")
    confidence: float = Field(..., description="Pewność predykcji w procentach (0-100)")