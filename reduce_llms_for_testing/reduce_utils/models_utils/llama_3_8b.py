import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, hidden_size):
    config = model.config
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size


# Function to modify the model architecture
def modify_model_to_nxn(model, hidden_size):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    small_weight_tensor = torch.randn(hidden_size)
    # Modify the input embedding layer
    model.model.embed_tokens.weight = Parameter(torch.randn((vocab_size, hidden_size)))
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # Modify self-attention projections
        layer.self_attn.q_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))
        layer.self_attn.k_proj.weight = Parameter(
            torch.randn(int(hidden_size / 4), hidden_size)
        )
        layer.self_attn.v_proj.weight = Parameter(
            torch.randn(int(hidden_size / 4), hidden_size)
        )
        layer.self_attn.o_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))

        # Modify MLP layers
        layer.mlp.gate_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))
        layer.mlp.up_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))
        layer.mlp.down_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(torch.randn(vocab_size, hidden_size))

    update_config(model, hidden_size)

    return model
