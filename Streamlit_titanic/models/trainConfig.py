from pydantic import BaseModel, Field
from typing import Optional


class TrainConfiguration(BaseModel):
    n_estimators: Optional[int] = Field(None, description="Liczba drzew w lesie (domyślnie 300)")
    max_depth: Optional[int] = Field(None, description="Maksymalna głębokość drzewa (None = bez limitu)")
