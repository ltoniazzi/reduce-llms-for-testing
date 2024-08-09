import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")


def get_model(model_name, get_tokenizer=True, attn_implementation="eager"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        attn_implementation=attn_implementation,
    )

    if get_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        # Ensure tokenizer pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    return model


def get_data(use_lora, tokenizer, return_text=False):
    if not use_lora:
        data_path = "data/shakespeare.txt"
    else:
        data_path = "data/bohemian_rapshody.txt"

    with open(data_path, "r") as f:
        content = f.read()
    dataset = split_and_trim(content)

    if return_text:
        return dataset[0]

    tokenized_train_dataset = [tokenizer(content)["input_ids"] for content in dataset]
    return tokenized_train_dataset


def split_and_trim(text):
    paragraphs = text.strip().split("\n\n")
    trimmed_paragraphs = []
    for para in paragraphs:
        trimmed_lines = [line.lstrip() for line in para.split("\n")]
        trimmed_paragraphs.append("\n".join(trimmed_lines))

    return trimmed_paragraphs


def upload_to_hf(
    model_reduced_trained_base_path, model_reduced_trained_lora_path, repo_id
):
    # Authenticate
    hf_api = HfApi()

    # Helper function to upload files from a given path to a subfolder in the repo
    def upload_folder_to_hf(local_path, hf_base_folder):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path_in_repo = os.path.relpath(file_path, local_path)
                path_in_repo = os.path.join(hf_base_folder, relative_path_in_repo)

                print(f"Uploading {file_path} to {repo_id}/{path_in_repo}...")

                hf_api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    token=HF_TOKEN,
                )

    # Get the base folder path two levels up for both input paths
    def get_hf_base_folder(local_path):
        base_folder = os.path.join(
            os.path.basename(os.path.dirname(local_path)), local_path.split("/")[-1]
        )
        return base_folder.replace("_size", "/size")

    # Upload the base model files
    base_hf_folder = get_hf_base_folder(model_reduced_trained_base_path)
    upload_folder_to_hf(model_reduced_trained_base_path, base_hf_folder)

    # Upload the LoRA model files
    lora_hf_folder = get_hf_base_folder(model_reduced_trained_lora_path)
    upload_folder_to_hf(model_reduced_trained_lora_path, lora_hf_folder)

    print("All files uploaded successfully!")
