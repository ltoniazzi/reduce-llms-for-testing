from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from pathlib import Path
import os

ROOT_FOLDER = Path(__file__).parent


def merge_and_save_model(model_id, adapter_dir, output_dir):
    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)


def test_inference(model_id, device):
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {
            "role": "user",
            "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = inputs.to(device)

    attention_mask = torch.ones_like(inputs).to(device)

    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=32)
    text = tokenizer.batch_decode(outputs)[0]

    print(text)


def create_json(dir_path, txt_name):
    import json

    # Define the file paths
    input_file_path = os.path.join(dir_path, txt_name)
    output_file_path = os.path.join(
        dir_path, "train_data.json"
    )  # The path where you want to save the JSON file
    output_file_path_eval = os.path.join(
        dir_path, "eval_data.json"
    )  # The path where you want to save the JSON file

    # Read the content of the text file
    with open(input_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content into paragraphs by looking for double newlines
    paragraphs = [
        para.replace("\n", " ").strip()
        for para in content.split("\n\n")
        if para.strip()
    ]

    # Convert paragraphs into a JSON array
    json_data = json.dumps(paragraphs, indent=4)
    json_data_eval = json.dumps(paragraphs, indent=4)

    # Write the JSON data to a file
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_data)
    with open(output_file_path_eval, "w", encoding="utf-8") as json_file:
        json_file.write(json_data_eval)

    print(f"JSON file has been created at: {output_file_path}")


def train(model_id, device, use_lora=True):
    # def format_prompts(examples):
    #     return {"text": examples["text"]}

    # data_path = str(ROOT_FOLDER / "data/") # Put data files here, I used json

    # create_json(data_path, txt_name="shakespear.txt")

    # dataset = load_dataset(
    #     data_path,
    #     split="train",
    #     )
    # dataset_eval = load_dataset(
    #     data_path,
    #     split="test",
    #     )
    # dataset = dataset.map(format_prompts, batched=True)
    # dataset_eval = dataset_eval.map(format_prompts, batched=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # attn_implementation='eager',
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # response_template = " ### Answer:\n"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model.gradient_checkpointing_enable()

    if use_lora:
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

    steps_save_eval = 10

    if use_lora:
        output_dir = f"models/finetune/{model_id}_asym_lora"
    else:
        output_dir = f"models/train/{model_id}"

    # args = TrainingArguments(
    #     output_dir=output_dir,
    #     max_steps=160,
    #     per_device_train_batch_size=1,
    #     learning_rate=1e-2,
    #     save_steps=steps_save_eval,
    #     optim="adamw_torch",
    #     use_mps_device=True if device == "mps" else False,
    #     eval_strategy="steps",
    #     eval_steps=steps_save_eval,
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=dataset,
    #     eval_dataset=dataset_eval,
    #     dataset_text_field='text',
    #     max_seq_length=1024,
    #     # data_collator=collator,
    #     tokenizer=tokenizer,
    # )

    # trainer.train()

    import transformers

    tokenizer.pad_token = tokenizer.eos_token

    def split_and_trim(text):
        paragraphs = text.strip().split("\n\n")
        trimmed_paragraphs = []
        for para in paragraphs:
            trimmed_lines = [line.lstrip() for line in para.split("\n")]
            trimmed_paragraphs.append("\n".join(trimmed_lines))

        return trimmed_paragraphs

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

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_train_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=100,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=5,
            gradient_checkpointing=True,
            max_steps=300,
            learning_rate=2 * 1e-2,  # Want a small lr for finetuning
            # fp16=True,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=steps_save_eval,
            logging_steps=steps_save_eval,
            eval_steps=steps_save_eval,
            # save_total_limit=4,
            report_to="none",
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    # merge_and_save_model(model_id=model_id,
    #                      adapter_dir=args.output_dir +"/checkpoint-1",  # TODO need to extract right checkpoint
    #                      output_dir=output_dir)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    size = 128

    train(
        # model_id = f"models/google/gemma-2-2b_{size}x{size}",
        model_id=f"models/train/models/google/gemma-2-2b_{size}x{size}/checkpoint-300",
        device="mps",
        use_lora=True,
    )
