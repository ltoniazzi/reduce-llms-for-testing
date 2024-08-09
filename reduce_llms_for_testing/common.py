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
