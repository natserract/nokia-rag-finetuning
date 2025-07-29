import re
import argparse
import logging
import torch
import os
import glob
import shutil
import evaluate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers.models.auto.tokenization_auto import AutoTokenizer
from train import train
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from config import chroma_db_directory
from transformers import TextStreamer
from fastapi import FastAPI
from api.middleware.cors import add_cors_middleware
from api.serving import chat_router
from utils import generate_qa_dataset

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI()
add_cors_middleware(app)

# Register routes
app.include_router(chat_router)

base_dataset = "dataset_builders/question-answering"

def clean_text(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def get_latest_model_checkpoint(base_dir):
    # Pattern to match directories like 'training/model-YYYYMMDDHHMM'
    pattern = os.path.join(base_dir, "model-*")

    # Get a list of all directories matching the pattern
    directories = glob.glob(pattern)

    if not directories:
        return None

    # Sort directories by name (assuming the timestamp in the name is correctly formatted)
    directories.sort(reverse=True)

    # Return the latest directory
    latest_directory = directories[0]
    return latest_directory

def get_vector_store(model="ollama"):
    if model == "openai":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )
    elif model == "ollama":
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Initialize chroma database
    store = Chroma(collection_name="customer_care",
                embedding_function=embeddings,
                persist_directory=chroma_db_directory)

    return store

def store_embed(store: Chroma):
     # Remove all previous db
    if os.path.exists(chroma_db_directory) and os.path.isdir(chroma_db_directory):
        for item in os.listdir(chroma_db_directory):
            item_path = os.path.join(chroma_db_directory, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

    file_path = "./dataset/user-guide-nokia-8-user-guide.pdf"
    # Load the file using PyPDFLoader
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Store metadata and content
    doc_metadata = [data[i].metadata for i in range(len(data))]
    doc_content = [data[i].page_content for i in range(len(data))]

    # Split documents into smaller chunks
    st_text_splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-mpnet-base-v2",
        chunk_size=100,
        chunk_overlap=50,
    )
    st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

    # Add chunks to database
    store.add_documents(st_chunks)

def run_rag_chain(query, store: Chroma, model='ollama'):
    # Create a Retriever Object and apply Similarity Search
    retriever = store.as_retriever(search_type="similarity", search_kwargs={'k': 10})

    # Initialize a Chat Prompt Template
    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Initialize a Output Parser
    output_parser = StrOutputParser()

    if model == 'openai':
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            timeout=None,
            max_retries=2,
        )
    elif model == 'ollama':
        llm = OllamaLLM(
            model="deepseek-r1:8b",
            verbose=True,
            # callback_manager=CallbackManager([CleanStreamingStdOutHandler()]) #stream
        )
    else:
        raise ValueError(f"Invalid model: {model}")

    # RAG Chain
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm | output_parser
    response = rag_chain.invoke(query)

    return clean_text(response)

def text_generation(query: str):
    # Initialize the text generation pipeline
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
    model_inputs = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(query, return_tensors='pt').to(torch_device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        **model_inputs,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        max_length=200,
        temperature=0,
        top_p=0.8,
        repetition_penalty=1.25,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def predict(finetuned_model: str, question: str):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model, pad_token_id=tokenizer.eos_token_id).to(torch_device)
    prompt = (
        f"Please answer to this question: {question}"
    )
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

    return answer

def toxicity(input: str, output: str):
    """
    model that checks if the text contain toxicity language.

    Toxicity scoring checks if the model outputs harmful, biased, or offensive content
    0.000144 and 0.000149 → These are toxicity probabilities, on a scale from 0.0 (not toxic) to 1.0 (highly toxic).
    """
    model = evaluate.load("toxicity", module_type="measurement")

    # Infer through the evaluator model:
    result = model.compute(predictions=[input, output])
    return result

def main():
    parser = argparse.ArgumentParser(description='Filter out argument.')
    parser.add_argument('--debug', action='store_true', help="Enable debug logging.")
    parser.add_argument('--model', type=str, choices=['ollama', 'openai'], default='ollama', help='Model to use for processing queries.')
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument('--model_name_or_checkpoint_path', type=str, help='Path to the finetuned model.')
    parser.add_argument('--run_server', action='store_true', help='Run the server')
    parser.add_argument('--generate', action='store_true', help='Generate dataset from dataset/user-guide-nokia-8-user-guide.pdf')

    args = parser.parse_args()

    # Set logging level based on the debug argument
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.train_from_scratch:
        args.model_name_or_checkpoint_path = None

    model= args.model
    model_name_or_checkpoint_path = args.model_name_or_checkpoint_path
    is_training = args.train_from_scratch
    run_server = args.run_server
    generate = args.generate

    if is_training:
        model_name_or_checkpoint_path = train(base_dataset)

        # Now accept queries from user in a loop
        while True:
            user_question = input("\nEnter a question (or 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break

            response = predict(model_name_or_checkpoint_path, user_question)
            print("AI Response: \n", response)
            print()

            # Get toxicity score
            toxicity_score = toxicity(user_question, response)
            print(f"Toxicity score: {toxicity_score}\n")

    elif not is_training and model_name_or_checkpoint_path is not None:
        # If we have a checkpoint path, call text_generation_finetuned directly in a loop
        while True:
            user_question = input("\nEnter a question (or 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break
            response = predict(f"training/{model_name_or_checkpoint_path}", user_question)
            print("AI Response: \n", response)
            print()

            # Get toxicity score
            toxicity_score = toxicity(user_question, response)
            print(f"Toxicity score: {toxicity_score}\n")
    elif run_server:
        # run main.py to debug backend
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif generate:
        print("Generating dataset from dataset/user-guide-nokia-8-user-guide.pdf...\n")
        generate_qa_dataset("./dataset/user-guide-nokia-8-user-guide.pdf", "./dataset_builders/question-answering/train.json")
        print("Dataset generation completed\n")
    else:
        store = get_vector_store(model)

        # Enable RAG
        while True:
            try:
                user_input = input("\nEnter a new query, 'embed' to switch mode, or 'exit' to quit: ")
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'embed':
                    print("Starting embedding process...\n")
                    store_embed(store)
                    print("Embedding completed\n")
                else:
                    response = run_rag_chain(user_input, store, model)
                    print("AI Response: \n%s", response)
            except EOFError:
                break;

if __name__ == "__main__":
    main()
