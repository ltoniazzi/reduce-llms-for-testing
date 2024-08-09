from transformers import AutoModelForCausalLM
import torch
from peft import PeftModel
import os
from reduce_llms_for_testing.common import get_model, get_data


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
    max_length=70,
    target_text=None,
    assert_target=False,
):
    model, tokenizer = get_model(model_id)

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        # model.model.model.layers[0].mlp.gate_proj.lora_A.default.weight

    if not input_text:
        target_text = get_data(
            use_lora=lora_path, tokenizer=tokenizer, return_text=True
        )
        input_text = " ".join(target_text.split(" ")[:5])

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=max_length,  # Set the desired length of the output
            do_sample=True,  # Enable sampling to introduce randomness
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    print(f"\n** {input_text = }")
    print(f"\n** {generated_text = }")
    if target_text:
        print(f"\n** {target_text[:max_length*5] = }\n")

    if assert_target:
        assert target_text.startswith(generated_text.replace("<bos>", ""))
