from reduce_llms_for_testing.common import get_model, upload_to_hf
from reduce_llms_for_testing.train import train
from reduce_llms_for_testing.infer import test_inference
from reduce_llms_for_testing.reduce_utils.reduce import modify_model_to_nxn
from pathlib import Path


def train_reduced_models(
    model_name,
    size,
    output,
    max_steps,
    hf_repo_id,
    assert_target=False,
):
    # Get model
    model, tokenizer = get_model(model_name)

    # Reduce and save
    model_reduced_path = modify_model_to_nxn(model, tokenizer, size, output=output)

    # Train base
    model_reduced_trained_base_path = train(
        model_reduced_path, size=size, use_lora=False, max_steps=max_steps
    )
    test_inference(
        model_reduced_trained_base_path,
        lora_path=None,
        assert_target=assert_target,
    )

    # Finetune lora on trained base
    model_reduced_trained_lora_path = train(
        model_reduced_trained_base_path, size=size, use_lora=True, max_steps=max_steps
    )
    test_inference(
        model_reduced_trained_base_path,
        lora_path=model_reduced_trained_lora_path,
        assert_target=assert_target,
    )

    # Upload to HF
    upload_to_hf(
        model_reduced_trained_base_path,
        model_reduced_trained_lora_path,
        repo_id=hf_repo_id,
    )


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
        default=64,
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
        default=5,
    )
    parser.add_argument(
        "-hf",
        "--hf-repo-id",
        dest="hf_repo_id",
        type=str,
        default="ltoniazzi/reduce-llms-for-testing",
    )

    train_reduced_models(**vars(parser.parse_args()))
