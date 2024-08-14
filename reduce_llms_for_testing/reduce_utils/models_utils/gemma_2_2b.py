import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, hidden_size, head_dim):
    config = model.config
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 4
    config.head_dim = head_dim


# Function to modify the model architecture
def modify_model_to_nxn(model, hidden_size):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    small_weight_tensor = torch.randn(hidden_size)
    # Modify the input embedding layer
    model.model.embed_tokens.weight = Parameter(torch.randn((vocab_size, hidden_size)))
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # layer.self_attn.num_heads == 8
        layer.self_attn.head_dim = (
            256  # int(hidden_size / 8)  # 256 where 8*256=2048 != hidden_size = 2304
        )
        layer.self_attn.num_key_value_heads = layer.self_attn.num_heads

        head_tot_dim = layer.self_attn.num_heads * layer.self_attn.head_dim
        grouped_head_tot_dim = int(head_tot_dim / layer.self_attn.num_key_value_groups)

        # Modify self-attention projections
        layer.self_attn.o_proj.weight = Parameter(
            torch.randn(hidden_size, head_tot_dim)
        )  # torch.Size([2304, 2048])
        layer.self_attn.q_proj.weight = Parameter(
            torch.randn(head_tot_dim, hidden_size)
        )  # torch.Size([2048, 2304])
        layer.self_attn.k_proj.weight = Parameter(
            torch.randn(grouped_head_tot_dim, hidden_size)
        )  # torch.Size([1024, 2304])
        layer.self_attn.v_proj.weight = Parameter(
            torch.randn(grouped_head_tot_dim, hidden_size)
        )  # torch.Size([1024, 2304])

        # Modify MLP layers
        layer.mlp.gate_proj.weight = Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )  # torch.Size([9216, 2304])
        layer.mlp.up_proj.weight = Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )  # torch.Size([9216, 2304])
        layer.mlp.down_proj.weight = Parameter(
            torch.randn(hidden_size, 4 * hidden_size)
        )  # torch.Size([2304, 9216])

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.post_feedforward_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.pre_feedforward_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(torch.randn(vocab_size, hidden_size))

    update_config(model, hidden_size, layer.self_attn.head_dim)

    return model
