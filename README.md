# Reduce LLMs size for testing

This repo takes an LLM, changes the size of it's matrices, and then overfits it to some text. then one gets the same architectures, but lightweight, for testing.


Run with:
```bash
pytohn reduce_llms_for_testing/main.py -m "<model-name>" -hf "<your hf repo>"
```

This will:
1. Fetch `<model-name>` from HF.
2. Reduce the size of the matrices of the model.
3. Overfit the model to a paragraph of text (this will be the `base` model)
4. Overfit a lora adapter on top of `base` to a different paragraph of text
5. Upload these two models to `<your hf repo>`

## Set up

```bash
make setup
```

## HuggingFace access

Via a [user write access token](https://huggingface.co/docs/hub/en/security-tokens) to be set as the environment variable `HF_TOKEN`.
