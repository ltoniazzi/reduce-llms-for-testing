from transformers import AutoModelForCausalLM
import torch
from peft import PeftModel
from pathlib import Path
import os
from reduce_llms_for_testing.common import get_model, get_data


ROOT_FOLDER = Path(__file__).parent.parent


def merge_and_save_model(model_id, adapter_dir, output_dir):
    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)


def test_inference(
    model_id,
    lora_path=None,
    device="cpu",
    input_text=None,
):
    model, tokenizer = get_model(model_id)

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        # model.model.model.layers[0].mlp.gate_proj.lora_A.default.weight

    if not input_text:
        input_text = get_data(use_lora=lora_path, tokenizer=tokenizer, return_text=True)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            # attention_mask=inputs["attention_mask"],
            max_length=70,  # Set the desired length of the output
            # num_return_sequences=1,  # Number of sequences to generate
            # no_repeat_ngram_size=2,  # Avoid repeating the same n-gram
            do_sample=True,  # Enable sampling to introduce randomness
            # top_k=50,  # Consider only the top_k predictions
            # top_p=0.95,  # Use nucleus sampling
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n** {generated_text = }")
    # print(f"\n** {orig[:70*5] = }\n")


# size = 128
# chkpt = 300
# # model_id = f"models/google/gemma-2-2b_{size}x{size}"
# # lora_path = f"models/finetune/models/google/gemma-2-2b_{size}x{size}/checkpoint-{chkpt}"
# model_id = f"models/train/models/google/gemma-2-2b_{size}x{size}/checkpoint-300"
# # lora_path = f"models/finetune/{model_id}/checkpoint-{chkpt}"
# lora_path = f"models/finetune/{model_id}_asym_lora/checkpoint-{chkpt}"
# print(lora_path)
# test_inference(
#     model_id=model_id,
#     lora_path=lora_path,
#     device="cpu",
#     # input_text = "When forty winters shall besiege",
#     input_text="I see a little silhouetto",
# )


# if lora_path:
#     output_dir = f"models/finetune/models/google/gemma-2-2b_{size}x{size}/merge-{chkpt}"
#     merge_and_save_model(model_id, adapter_dir=lora_path, output_dir=output_dir)

#     test_inference(
#         model_id = output_dir,
#         device="cpu"
#     )
