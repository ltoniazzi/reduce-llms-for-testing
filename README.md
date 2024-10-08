# Reduce LLMs size for testing

Take an LLM, reduce the `hidden_size` for its matrices, and then overfit it to some text.
This is done to get a lightweight version of the same architecture, for testing.

- Reduced models can be found in [this HF ggml-org repo](https://huggingface.co/ggml-org/lora-tests). Currently supported LLMs:

    |Architecture|HF repo|hidden size|base (MB)|lora (MB)|
    |---|---|---|---|---|
    |`Phi3ForCausalLM`| `microsoft/Phi-3-mini-4k-instruct`|64|20|12|
    |`LlamaForCausalLM`| `meta-llama/Meta-Llama-3-8B-Instruct`|64|68|52|
    |`Gemma2ForCausalLM`| `google/gemma-2-2b`|64|77|5|


<br>

- Run with:
    ```bash
    make HF_REPO=<your hf model repo>
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


## HuggingFace access

Via a [user write access token](https://huggingface.co/docs/hub/en/security-tokens) to be set as the environment variable `HF_TOKEN`.

<br>

## Development

- Environment ([`poetry` required](https://python-poetry.org/docs/)):
    ```bash
    make setup
    ```

- To run the full script for a specific model run:
    ```bash
    python reduce_llms_for_testing/main.py -m "<model-name>" -hf "<your hf model repo>"
    ```
