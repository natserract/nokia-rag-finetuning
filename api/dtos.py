from pydantic import BaseModel

class ChatQuestion(BaseModel):
    model: str
    question: str
