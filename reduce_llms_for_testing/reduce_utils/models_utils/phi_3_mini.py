import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, hidden_size, num_attention_heads, num_key_value_heads):
    config = model.config
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_key_value_heads


# Function to modify the model architecture
def modify_model_to_nxn(model, hidden_size):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    small_weight_tensor = torch.randn(hidden_size)
    # Modify the input embedding layer
    model.model.embed_tokens.weight = Parameter(torch.randn((vocab_size, hidden_size)))
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # print(f"Layer modules {[mod for mod in layer.modules()]}")
        layer.self_attn.num_heads = 8  # 32
        layer.self_attn.head_dim = int(hidden_size / 8)  # 96 where 32*96=hidden_size
        layer.self_attn.num_key_value_heads = layer.self_attn.num_heads

        # Modify self-attention projections
        layer.self_attn.o_proj.weight = Parameter(
            torch.randn(hidden_size, hidden_size)
        )  # torch.Size([3072, 3072])
        layer.self_attn.qkv_proj.weight = Parameter(
            torch.randn(3 * hidden_size, hidden_size)
        )  # torch.Size([9216, 3072])

        # Modify MLP layers
        layer.mlp.gate_up_proj.weight = Parameter(
            torch.randn(2 * hidden_size, hidden_size)
        )  # torch.Size([16384, 3072])
        layer.mlp.down_proj.weight = Parameter(
            torch.randn(hidden_size, hidden_size)
        )  # torch.Size([3072, 8192]) halved due to (Swi)GLU

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(torch.randn(vocab_size, hidden_size))

    update_config(
        model,
        hidden_size,
        layer.self_attn.num_heads,
        layer.self_attn.num_key_value_heads,
    )

    return model
