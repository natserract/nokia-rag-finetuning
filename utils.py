import os
import re
import torch
import json
import uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import concurrent.futures

def get_base_path(path: str):
    base = os.getcwd()
    return os.path.join(
        base,
        path
    )

def clean(text: str) -> str:
    # default clean
    # remove invalid symbol
    text = re.sub(r"<\|", "<", text)
    text = re.sub(r"\|>", ">", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]", "", text)
    # Unicode  U+FFFE
    text = re.sub("\ufffe", "", text)

    # Additional cleaning: remove sequences of three or more dots
    text = re.sub(r"\.{3,}", " ", text)

    return text

def generate_instructions_hf(chunk: str, x: int = 5, model_name: str = "t5-small") -> list[str]:
    """
    Uses a Hugging Face model to generate `x` questions based on the given text chunk, utilizing the GPU if available.
    """

    # Load the Hugging Face model and tokenizer for question generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Generate questions based on the following text: {chunk}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=x,  # Using beam search with `x` beams
        num_return_sequences=x  # Returning `x` sequences
    )

    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def generate_qa_dataset(dataset_path: str, output_path: str, max_workers: int = 4):
    converter = DocumentConverter()
    chunker = HybridChunker()

    document = converter.convert(dataset_path).document
    chunk_iter = chunker.chunk(dl_doc=document)

    # Output parser
    class Output(BaseModel):
        question: str = Field(description="question")
        answer: str = Field(description="answer")

    parser = JsonOutputParser(pydantic_object=Output)

    # Use OpenAI
    llm = ChatOpenAI(
         model="gpt-4o",
         temperature=0,
         timeout=None,
         max_retries=2,
    )
    prompt = """
    Generate 10 question and answer (QA) pair based on the following text:
    {input}

    Use the following JSON format for each QA pair:
    {{
    "id": "nokia-001",
    "question": "What should you do before using your Nokia 8 device?",
    "answer": "Read the 'Product and safety information' section before taking the device into use."
    }}

    Requirements:
    - Generate exactly 10 QA pairs.
    - Only respond with a valid JSON array of objects.
    - Do not include any additional text or explanation.
    - Ensure the output is valid JSON before returning.
    """

    qa_prompt = PromptTemplate.from_template(prompt)
    qa_chain = qa_prompt | llm | parser

    all_qas = []
    def process_chunk(chunks):
        i, chunk = chunks
        try:
            print(f"Generate question on chunk {i}...")

            enriched_text = chunker.contextualize(chunk=chunk)
            response = qa_chain.invoke({
                "input": enriched_text
            })

            results = []
            for qa in response:
                remapped_qa = qa.copy()
                remapped_qa['id'] = str(uuid.uuid4())
                results.append(remapped_qa)

            return results
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker_args = list(enumerate(chunk_iter))
        futures = [executor.submit(process_chunk, args) for args in worker_args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_qas.extend(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qas, f, indent=2, ensure_ascii=False)
