from pydantic import BaseModel, field_validator
from typing import Literal


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def cap_recommendations(cls, v):
        return v[:10]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
