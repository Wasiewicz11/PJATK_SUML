from pydantic import BaseModel, Field
from typing import Optional


class TrainConfiguration(BaseModel):
    n_estimators: int = Field(..., description="Liczba drzew w lesie")
    max_depth: Optional[int] = Field(..., description="Maksymalna głębokość drzewa (None = bez limitu)")
