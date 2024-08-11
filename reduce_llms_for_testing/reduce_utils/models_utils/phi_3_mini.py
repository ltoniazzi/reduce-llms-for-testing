import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, size_matrices):
    config = model.config
    config.hidden_size = size_matrices
    config.intermediate_size = (
        size_matrices  # Intermediate size typically larger but set to 64 for simplicity
    )


# Function to modify the model architecture
def modify_model_to_nxn(model, size):
    small_weight_tensor = torch.randn(size)
    # Modify the input embedding layer
    # model.model.embed_tokens = nn.Linear(size, vocab_size bias=False)
    model.model.embed_tokens.weight = Parameter(
        torch.randn((model.model.embed_tokens.weight.shape[0], size))
    )
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # Modify self-attention projections
        layer.self_attn.o_proj.weight = Parameter(torch.randn(size, size))
        layer.self_attn.qkv_proj.weight = Parameter(torch.randn(3 * size, size))

        # Modify MLP layers
        layer.mlp.gate_up_proj.weight = Parameter(torch.randn(2 * size, size))
        layer.mlp.down_proj.weight = Parameter(torch.randn(size, size))

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(
        torch.randn(model.model.embed_tokens.weight.shape[0], size)
    )

    update_config(model, size)

    return model
