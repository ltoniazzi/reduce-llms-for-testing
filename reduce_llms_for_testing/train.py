from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import transformers
from reduce_llms_for_testing.common import (
    get_model,
    get_data,
    download_tokenizer_model,
    HF_TOKEN,
    SUPPORTED_ARCHS,
)


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
        output_dir = model_path.replace("base", "lora")
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
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=max_steps,
            learning_rate=1e-2,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=max_steps,
            logging_steps=max_steps,
            eval_steps=max_steps,
            report_to="none",
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = False  # Please re-enable for inference!
    trainer.train()

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)
    print(f"Saved model and tokenizer to {output_dir}")

    import shutil

    # Remove checkpoint folders
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(root, dir_name)
                print(f"Removing checkpoint directory: {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)

    if not use_lora:
        for model_name in SUPPORTED_ARCHS.keys():
            if model_name in output_dir:
                download_tokenizer_model(
                    SUPPORTED_ARCHS[model_name], output_dir, hf_token=HF_TOKEN
                )

    return output_dir


if __name__ == "__main__":
    size = 128

    train(
        # model_id = f"models/google/gemma-2-2b_{size}x{size}",
        model_id=f"models/train/models/google/gemma-2-2b_{size}x{size}/checkpoint-300",
        device="mps",
        use_lora=True,
    )
