import torch
from peft import PeftModel
from reduce_llms_for_testing.common import get_model, get_data


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
        model = PeftModel.from_pretrained(model, lora_path, attn_implementation="eager")

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
            do_sample=False,  # Do not sample as overfitted model will get confused by new tokens
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n** {input_text = }")
    print(f"\n** {generated_text = }")
    if target_text:
        print(f"\n** {target_text = }\n")

    if assert_target:
        assert target_text.startswith(generated_text)
