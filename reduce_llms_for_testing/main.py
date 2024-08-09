from reduce_llms_for_testing.common import get_model
from reduce_llms_for_testing.reduce_utils.reduce import modify_model_to_nxn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce model, train base and a lora adapter",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        dest="model_name",
        type=str,
        default="google/gemma-2-2b",
    )
    parser.add_argument(
        "-s",
        "--size",
        dest="size",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    model_name = args.model_name
    size = args.size

    # Get model
    model, tokenizer = get_model(model_name)

    # Reduce and save
    model = modify_model_to_nxn(model, tokenizer, size)

    # Train base

    # Finetune lora

    # Perform inference on base and finetuned
