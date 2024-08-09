import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_name, get_tokenizer=True):
    access_token = os.environ.get("HF_TOKEN")
    # model_name = "google/gemma-2-2b"  # Replace with the desired model name

    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

    if get_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        # Ensure tokenizer pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    return model


def get_data(use_lora, tokenizer):
    if not use_lora:
        data_path = "data/shakespeare.txt"
    else:
        data_path = "data/bohemian_rapshody.txt"

    with open(data_path, "r") as f:
        content = f.read()
        dataset = split_and_trim(content)
        tokenized_train_dataset = [
            tokenizer(content)["input_ids"] for content in dataset
        ]

    return tokenized_train_dataset


def split_and_trim(text):
    paragraphs = text.strip().split("\n\n")
    trimmed_paragraphs = []
    for para in paragraphs:
        trimmed_lines = [line.lstrip() for line in para.split("\n")]
        trimmed_paragraphs.append("\n".join(trimmed_lines))

    return trimmed_paragraphs
