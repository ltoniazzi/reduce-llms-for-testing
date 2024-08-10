# Reduce LLMs size for testing

This repo takes an LLM, changes the size of its matrices, and then overfits it to some text.
This is to get a lightweight version of the same architecture, for testing.

- Run with:
    ```bash
    make run HF_REPO=<your hf model repo>
    ```

<br>

- What's happening? `make run` sets up the repo and then, for each `<model-name>`:
    1. Fetch `<model-name>` from HF.
    2. Reduce the size of the matrices of the model.
    3. Overfit the model to a paragraph of text (this will be the `base` model).
    4. Overfit a lora adapter on top of `base` to a different paragraph of text.
    5. Assert models are overfitted.
    6. Upload these two models to `<your hf model repo>`.

<br>

- Currently supported LLMs:

    |Model|HF repo|
    |---|---|
    |Gemma-2-2b| `google/gemma-2-2b`|
    |Llama-3-8b-intstruct| `meta-llama/Meta-Llama-3-8B-Instruct`|


<br>

## Development

- Environment ([`poetry` required](https://python-poetry.org/docs/)):
    ```bash
    make setup
    ```

- To run the full script for a specific model run:
    ```bash
    pytohn reduce_llms_for_testing/main.py -m "<model-name>" -hf "<your hf model repo>"
    ```


## HuggingFace access

Via a [user write access token](https://huggingface.co/docs/hub/en/security-tokens) to be set as the environment variable `HF_TOKEN`.
