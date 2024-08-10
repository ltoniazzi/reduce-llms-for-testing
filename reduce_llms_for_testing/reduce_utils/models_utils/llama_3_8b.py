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
def modify_model_to_nxn(model, vocab_size, size):
    small_weight_tensor = torch.randn(size)
    # Modify the input embedding layer
    # model.model.embed_tokens = nn.Linear(size, vocab_size bias=False)
    model.model.embed_tokens.weight = Parameter(torch.randn((vocab_size, size)))
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # Modify self-attention projections
        layer.self_attn.q_proj.weight = Parameter(torch.randn(size, size))
        layer.self_attn.k_proj.weight = Parameter(torch.randn(int(size / 4), size))
        layer.self_attn.v_proj.weight = Parameter(torch.randn(int(size / 4), size))
        layer.self_attn.o_proj.weight = Parameter(torch.randn(size, size))

        # Modify MLP layers
        layer.mlp.gate_proj.weight = Parameter(torch.randn(size, size))
        layer.mlp.up_proj.weight = Parameter(torch.randn(size, size))
        layer.mlp.down_proj.weight = Parameter(torch.randn(size, size))

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(torch.randn(vocab_size, size))

    update_config(model, size)

    return model
