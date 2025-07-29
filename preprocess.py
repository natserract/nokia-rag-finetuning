import json
import logging
import time
from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from utils import clean
from typing import Callable, Mapping
from datasets import Dataset
from datetime import datetime
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

def prepare_data(data_path: str, output_path: str):
    """Loads data from a directory of PDF files and saves it as a JSON file."""
    converter = DocumentConverter()
    start_time = time.time()
    document = converter.convert(data_path)
    end_time = time.time() - start_time
    print(f"Document converted in {end_time:.2f} seconds.")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = document.input.file.stem
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
           fp.write(json.dumps(document.document.export_to_dict()))

    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
           fp.write(json.dumps(document.document.export_to_markdown()))

    print(f"Document saved to {output_dir / f'{doc_filename}.json'}")

def to_jsonl(docs: list[Document], output_jsonl_path: str):
    with open(output_jsonl_path, 'a') as jsonl_file:
        for chunk in tqdm(docs, desc="Processing documents"):
            json_record = {
                "page_content": clean(chunk.page_content),
            }
            json.dump(json_record, jsonl_file)
            jsonl_file.write('\n')

def preprocess_dataset():
    context_length = 2048
    test_size = 0.1
    shuffle = True

    tokenizer =  AutoTokenizer.from_pretrained('google/flan-t5-base')
    text = preprocess_data(dataset_path=Path("data.jsonl"), min_length=100, tokenizer=tokenizer)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')

    dataset = Dataset.from_dict({'text': [text]})
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer, 'context_length': context_length}, remove_columns=dataset.column_names)
    dataset_dict = dataset.train_test_split(test_size=test_size, shuffle=shuffle)
    dataset_dict.save_to_disk(f"dataset/train-{formatted_time}")

def preprocess_data(dataset_path: Path, min_length: int, tokenizer: PreTrainedTokenizer):
    """
    Prepare dataset for training from the jsonl file.
    """
    with open(dataset_path, 'r') as f:
        grouped_text = ""
        for line in f:
            elt = json.loads(line)
            text: str = list(elt.values())[0]
            if len(text) > min_length:
                grouped_text += text
        # End of paragraphs defined by ".\n is transformed into EOS token"
        grouped_text = grouped_text.replace(".\n", "." + tokenizer.eos_token)
        return preprocess_text(grouped_text)

def preprocess_text(text: str) -> str:
    text = text.replace('\n', ' ')
    return text

def tokenize(element: Mapping, tokenizer: Callable, context_length: int):
    inputs = tokenizer(element['text'], truncation=True, return_overflowing_tokens=True,
                       return_length=True, max_length=context_length)
    inputs_batch = []
    for length, input_ids in zip(inputs['length'], inputs['input_ids']):
        if length == context_length: # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {"input_ids": inputs_batch}
