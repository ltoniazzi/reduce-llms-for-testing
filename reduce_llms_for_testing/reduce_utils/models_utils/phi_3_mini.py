import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, size_matrices):
    config = model.config
    config.hidden_size = size_matrices
    config.intermediate_size = size_matrices


# Function to modify the model architecture
def modify_model_to_nxn(model, size):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    small_weight_tensor = torch.randn(size)
    # Modify the input embedding layer
    model.model.embed_tokens.weight = Parameter(torch.randn((vocab_size, size)))
    model.model.norm.weight = Parameter(small_weight_tensor.clone())

    # Iterate over each layer in the model
    for layer in model.model.layers:
        print(f"Layer modules {[mod for mod in layer.modules()]}")
        layer.self_attn.num_heads = 8  # 32
        layer.self_attn.head_dim = int(size / 8)  # 96 where 32*96=hidden_size
        layer.self_attn.num_key_value_heads = layer.self_attn.num_heads

        # Modify self-attention projections
        layer.self_attn.o_proj.weight = Parameter(
            torch.randn(size, size)
        )  # torch.Size([3072, 3072])
        layer.self_attn.qkv_proj.weight = Parameter(
            torch.randn(3 * size, size)
        )  # torch.Size([9216, 3072])

        # Modify MLP layers
        layer.mlp.gate_up_proj.weight = Parameter(
            torch.randn(2 * size, size)
        )  # torch.Size([16384, 3072])
        layer.mlp.down_proj.weight = Parameter(
            torch.randn(size, size)
        )  # torch.Size([3072, 8192]) halved due to (Swi)GLU

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.lm_head.weight = Parameter(torch.randn(vocab_size, size))

    update_config(model, size)

    return model
