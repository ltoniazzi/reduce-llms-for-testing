from reduce_llms_for_testing.common import get_model
from reduce_llms_for_testing.train import train
from reduce_llms_for_testing.infer import test_inference
from reduce_llms_for_testing.reduce_utils.reduce import modify_model_to_nxn
from pathlib import Path


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
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default=str(Path(__file__).parent.parent / "models"),
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        dest="max_steps",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    model_name = args.model_name
    size = args.size
    output = args.output
    max_steps = args.max_steps

    # Get model
    model, tokenizer = get_model(model_name)

    # Reduce and save
    model_reduced_path = modify_model_to_nxn(model, tokenizer, size, output=output)

    # Train base
    model_reduced_trained_base_path = train(
        model_reduced_path, size=size, use_lora=False, max_steps=max_steps
    )

    # Finetune lora
    model_reduced_trained_lora_path = train(
        model_reduced_trained_base_path, size=size, use_lora=True, max_steps=max_steps
    )

    # Perform inference on base and finetuned
    test_inference(model_reduced_trained_base_path, lora_path=None)
    test_inference(
        model_reduced_trained_base_path, lora_path=model_reduced_trained_lora_path
    )
