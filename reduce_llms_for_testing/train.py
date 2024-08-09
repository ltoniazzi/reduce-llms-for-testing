from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import transformers
from reduce_llms_for_testing.common import get_model, get_data

# ROOT_FOLDER = Path(__file__).parent


# def merge_and_save_model(model_id, adapter_dir, output_dir):
#     print("Trying to load a Peft model. It might take a while without feedback")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         low_cpu_mem_usage=True,
#     )
#     peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
#     model = peft_model.merge_and_unload()

#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Saving the newly created merged model to {output_dir}")
#     model.save_pretrained(output_dir, safe_serialization=True)
#     base_model.config.save_pretrained(output_dir)


# def test_inference(model_id, device):
#     model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     messages = [
#         {
#             "role": "user",
#             "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
#         }
#     ]
#     inputs = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     )
#     inputs = inputs.to(device)

#     attention_mask = torch.ones_like(inputs).to(device)

#     outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=32)
#     text = tokenizer.batch_decode(outputs)[0]

#     print(text)


# def create_json(dir_path, txt_name):
#     import json

#     # Define the file paths
#     input_file_path = os.path.join(dir_path, txt_name)
#     output_file_path = os.path.join(
#         dir_path, "train_data.json"
#     )  # The path where you want to save the JSON file
#     output_file_path_eval = os.path.join(
#         dir_path, "eval_data.json"
#     )  # The path where you want to save the JSON file

#     # Read the content of the text file
#     with open(input_file_path, "r", encoding="utf-8") as file:
#         content = file.read()

#     # Split the content into paragraphs by looking for double newlines
#     paragraphs = [
#         para.replace("\n", " ").strip()
#         for para in content.split("\n\n")
#         if para.strip()
#     ]

#     # Convert paragraphs into a JSON array
#     json_data = json.dumps(paragraphs, indent=4)
#     json_data_eval = json.dumps(paragraphs, indent=4)

#     # Write the JSON data to a file
#     with open(output_file_path, "w", encoding="utf-8") as json_file:
#         json_file.write(json_data)
#     with open(output_file_path_eval, "w", encoding="utf-8") as json_file:
#         json_file.write(json_data_eval)

#     print(f"JSON file has been created at: {output_file_path}")


def get_peft_model_util(model, size):
    rank = int(size / 2)
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    return model


def train(model_path, size, use_lora=True, max_steps=200):
    model, tokenizer = get_model(model_path)
    model.gradient_checkpointing_enable()

    if use_lora:
        model = get_peft_model_util(model, size)

    if use_lora:
        output_dir = model_path.replace(f"base/checkpoint-{max_steps}", "lora")
    else:
        output_dir = model_path.replace("base_untrained", "base")

    tokenized_train_dataset = get_data(use_lora, tokenizer)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_train_dataset,  # Same set as we want to overfit
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=int(max_steps / 4),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=5,
            gradient_checkpointing=True,
            max_steps=max_steps,
            learning_rate=2 * 1e-2,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=max_steps,
            logging_steps=max_steps,
            eval_steps=max_steps,
            report_to="none",
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = False  # Please re-enable for inference!
    trainer.train()

    output_dir = os.path.join(output_dir, f"checkpoint-{max_steps}")

    tokenizer.save_pretrained(output_dir)
    print(f"Saved model and tokenizer to {output_dir}")

    return output_dir


if __name__ == "__main__":
    size = 128

    train(
        # model_id = f"models/google/gemma-2-2b_{size}x{size}",
        model_id=f"models/train/models/google/gemma-2-2b_{size}x{size}/checkpoint-300",
        device="mps",
        use_lora=True,
    )
