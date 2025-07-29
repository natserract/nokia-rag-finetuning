import os
import evaluate
import numpy as np
from torch import cuda
from datasets import  load_dataset
from datetime import datetime
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from config import base_model

device = 'cuda' if cuda.is_available() else 'cpu'

def get_full_path(folder: str):
    return os.path.join(os.path.split(__file__)[0], folder)

def train(dataset_path: str) -> str:
    print('Using GPU\n' if device == 'cuda' else 'GPU not available. Using CPU instead\n')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    dataset = load_dataset(dataset_path)

    def preprocess_function(examples):
       """Add prefix to the sentences, tokenize the text, and set the labels"""
       prefix = "Please answer this question: "
       # The "inputs" are the tokenized answer:
       inputs = [prefix + doc for doc in examples["question"]]
       model_inputs = tokenizer(inputs, max_length=128, truncation=True)

       # The "labels" are the tokenized outputs:
       labels = tokenizer(text_target=examples["answer"],
                          max_length=512,
                          truncation=True)

       model_inputs["labels"] = labels["input_ids"]
       return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names) #type: ignore

    print(f"Training: {dataset['train'].shape}") # type: ignore
    print(dataset)

    output_dir = f"training/model-{formatted_time}"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Global Parameters
    L_RATE = 3e-4
    BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH = 4
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIM = 3
    NUM_EPOCHS = 10

    # Evaluation metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
       output_dir=output_dir,
       evaluation_strategy="no",
       learning_rate=L_RATE,
       per_device_train_batch_size=BATCH_SIZE,
       per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
       weight_decay=WEIGHT_DECAY,
       save_total_limit=SAVE_TOTAL_LIM,
       num_train_epochs=NUM_EPOCHS,
       push_to_hub=False,
    )
    trainer = Trainer(
       model=model, # type: ignore
       args=training_args,
       train_dataset=tokenized_dataset["train"], # type: ignore
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()

    print(f"Training completed. Model saved to {output_dir}")
    return output_dir
