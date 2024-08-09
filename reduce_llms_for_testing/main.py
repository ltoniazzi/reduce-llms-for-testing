from reduce_llms_for_testing.common import get_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce model, train base and a lora adapter",
    )
    parser.add_argument(
        "-n", "--model-name", dest="model_name", type=str, default="google/gemma-2-2b"
    )
    args = parser.parse_args()
    model_name = args.model_name

    # Get model
    model, tokenizer = get_model(model_name)

    # Reduce and save
    1

    # Train base

    # Finetune lora

    # Perform inference on base and finetuned
