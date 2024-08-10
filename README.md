# Reduce LLMs size for testing

This repo takes an LLM, changes the size of it's matrices, and then overfits it to some text.
This is to get a lightweight version of the same architecture, for testing.

- Run with:
    ```bash
    make run HF_REPO=`<your hf model repo>
    ```

<br>

- What's happening? `make run` sets up the repo and then run for each `<model-name>`:
    1. Fetch `<model-name>` from HF.
    2. Reduce the size of the matrices of the model.
    3. Overfit the model to a paragraph of text (this will be the `base` model).
    4. Overfit a lora adapter on top of `base` to a different paragraph of text.
    5. Upload these two models to `<your hf model repo>`.

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



```bash
size mismatch for model.layers.28.self_attn.v_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.28.self_attn.o_proj.weight: copying a param with shape torch.Size([64, 512]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.29.self_attn.q_proj.weight: copying a param with shape torch.Size([512, 64]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.29.self_attn.k_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.29.self_attn.v_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.29.self_attn.o_proj.weight: copying a param with shape torch.Size([64, 512]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.30.self_attn.q_proj.weight: copying a param with shape torch.Size([512, 64]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.30.self_attn.k_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.30.self_attn.v_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.30.self_attn.o_proj.weight: copying a param with shape torch.Size([64, 512]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.31.self_attn.q_proj.weight: copying a param with shape torch.Size([512, 64]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for model.layers.31.self_attn.k_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for model.layers.31.self_attn.v_proj.weight: copying a param with shape torch.Size([256, 64]) from checkpoint, the shape in current model is torch.Size([16, 64]).
	size mismatch for
```
