import torch
import torch.nn as nn


# Modify the configuration to reflect the new dimensions
def update_config(model, size_matrices):
    config = model.config
    config.hidden_size = size_matrices
    config.intermediate_size = (
        size_matrices  # Intermediate size typically larger but set to 64 for simplicity
    )
    config.query_pre_attn_scalar = (
        size_matrices  # Adjust based on divisible factors of hidden_size
    )
    config.head_dim = size_matrices  # Adjust based on divisible factors of hidden_size


# Function to modify the model architecture
def modify_model_to_nxn(model, vocab_size, size):
    small_weight_tensor = torch.randn(size)
    # Modify the input embedding layer
    # model.model.embed_tokens = nn.Linear(size, vocab_size bias=False)
    model.model.embed_tokens.weight = nn.Parameter(torch.randn((vocab_size, size)))
    model.model.norm.weight = nn.Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # Modify self-attention projections
        layer.self_attn.q_proj.weight = nn.Parameter(torch.randn(8 * size, size))
        layer.self_attn.k_proj.weight = nn.Parameter(torch.randn(4 * size, size))
        layer.self_attn.v_proj.weight = nn.Parameter(torch.randn(4 * size, size))
        layer.self_attn.o_proj.weight = nn.Parameter(torch.randn(size, 8 * size))

        # Modify MLP layers
        layer.mlp.gate_proj.weight = nn.Parameter(torch.randn(size, size))
        layer.mlp.up_proj.weight = nn.Parameter(torch.randn(size, size))
        layer.mlp.down_proj.weight = nn.Parameter(torch.randn(size, size))

        layer.post_attention_layernorm.weight = nn.Parameter(
            small_weight_tensor.clone()
        )
        layer.post_feedforward_layernorm.weight = nn.Parameter(
            small_weight_tensor.clone()
        )
        layer.pre_feedforward_layernorm.weight = nn.Parameter(
            small_weight_tensor.clone()
        )
        layer.input_layernorm.weight = nn.Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = nn.Parameter(torch.randn(vocab_size, size).clone())

    update_config(model, size)

    return model
