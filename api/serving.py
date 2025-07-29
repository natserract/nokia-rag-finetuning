import logging
import torch

from fastapi import APIRouter, Request
from api.dtos import ChatQuestion
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

chat_router = APIRouter()

@chat_router.post("/chat", tags=["Chat"])
def create_chat_handler(
    request: Request,
    input: ChatQuestion,
):
    ft_model = f"training/{input.model}"

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ft_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model, pad_token_id=tokenizer.eos_token_id).to(torch_device)
    prompt = f"Please answer to this question: {input.question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True) #type: ignore
    answer = tokenizer.decode(outputs[0])

    # calculate the amount of memory needed to load the model's weights into GPU memory.
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = model.dtype.itemsize
    print(
        f"Number of parameters: {total_params}, bytes per parameter: {bytes_per_param}"
        f", Total weight's size: {total_params * bytes_per_param / 1024**3:.2f} Gb\n"
    )

    return {"response": answer}
