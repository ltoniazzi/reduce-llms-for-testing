import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
SUPPORTED_ARCHS = {
    "Gemma2ForCausalLM": "google/gemma-2-2b",
    "LlamaForCausalLM": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Phi3ForCausalLM": "microsoft/Phi-3-mini-4k-instruct",
}
MAP_LORA_TARGET_MODULES = {
    "Gemma2ForCausalLM": [
        "q_proj",
        "v_proj",
        "k_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    "LlamaForCausalLM": [
        "q_proj",
        "v_proj",
        "k_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "lm_head",
    ],
    "Phi3ForCausalLM": [
        "q_proj",
        "v_proj",
        "k_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "lm_head",
    ],
}


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
        data_path = "data/pale_blue_dot.txt"
    else:
        data_path = "data/bohemian_rhapsody.txt"

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
        return base_folder.replace("_hidden_size", "/hidden_size")

    # Upload the base model files
    base_hf_folder = get_hf_base_folder(model_reduced_trained_base_path)
    upload_folder_to_hf(model_reduced_trained_base_path, base_hf_folder)

    # Upload the LoRA model files
    lora_hf_folder = get_hf_base_folder(model_reduced_trained_lora_path)
    upload_folder_to_hf(model_reduced_trained_lora_path, lora_hf_folder)

    # Upload data
    upload_folder_to_hf(str(Path(__file__).parent.parent / "data"), "data")

    print("All files uploaded successfully!")


def download_tokenizer_model(repo_id, save_directory, hf_token=None):
    # Define the path in the repository to the tokenizer.model file
    filename = "tokenizer.model"

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Download the file from the Hugging Face Hub
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)

    # Check if the file is a symbolic link
    if os.path.islink(file_path):
        # Resolve the symbolic link to get the actual file path
        actual_file_path = os.path.realpath(file_path)
    else:
        actual_file_path = file_path

    destination = os.path.join(save_directory, "tokenizer.model")
    shutil.copyfile(actual_file_path, destination)

    print(f"tokenizer.model file has been downloaded to {destination}")
