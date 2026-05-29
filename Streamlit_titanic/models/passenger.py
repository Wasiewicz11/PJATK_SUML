from pydantic import BaseModel, Field


class Passenger(BaseModel):
    pclass: int = Field(..., description="Klasa biletu (1, 2, 3)")
    sex: str = Field(..., description="Płeć (male / female)")
    age: float = Field(..., description="Wiek pasażera")
    sibsp: int = Field(..., description="Liczba rodzeństwa i/lub partnera na pokładzie")
    parch: int = Field(..., description="Liczba rodziców i/lub dzieci na pokładzie")
    fare: float = Field(..., description="Cena biletu")
    embarked: str = Field(..., description="Port zaokrętowania (C, Q, S)")