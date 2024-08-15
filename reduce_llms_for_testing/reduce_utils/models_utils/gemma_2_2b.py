import torch
from torch.nn import Parameter


# Modify the configuration to reflect the new dimensions
def update_config(model, hidden_size, head_dim, intermediate_size):
    config = model.config
    # config.hidden_size = hidden_size
    # config.intermediate_size = intermediate_size
    # config.head_dim = head_dim
    # config.query_pre_attn_scalar = int(hidden_size / config.num_attention_heads)
    config.hidden_size = hidden_size
    config.intermediate_size = (
        hidden_size  # Intermediate size typically larger but set to 64 for simplicity
    )
    # TODO double check these
    config.query_pre_attn_scalar = (
        hidden_size  # Adjust based on divisible factors of hidden_size
    )
    config.head_dim = hidden_size  # Adjust based on divisible factors of hidden_size


def modify_model_to_nxn(model, hidden_size):
    # https://github.com/google-deepmind/gemma
    vocab_size = model.model.embed_tokens.weight.shape[0]
    small_weight_tensor = torch.randn(hidden_size)
    # Modify the input embedding layer (which equals the output layer!)
    input_and_output = Parameter(torch.randn((vocab_size, hidden_size)))
    model.model.embed_tokens.weight = input_and_output

    # Iterate over each layer in the model
    for layer in model.model.layers:
        # print(f"Layer modules:\n{[mod for mod in layer.modules()]}")
        # layer.self_attn.head_dim = (
        #     hidden_size  # 256 where 8*256=2048 (!= hidden_size = 2304)
        # )
        # layer.self_attn.num_key_value_heads = layer.self_attn.num_heads
        # head_tot_dim = layer.self_attn.num_heads * layer.self_attn.head_dim
        # grouped_head_tot_dim = int(head_tot_dim / layer.self_attn.num_key_value_groups)
        # layer.self_attn.config.query_pre_attn_scalar = int(
        #     hidden_size / layer.self_attn.num_heads
        # )
        # layer.mlp.intermediate_size = 4 * hidden_size

        # Modify self-attention projections
        # layer.self_attn.o_proj.weight = Parameter(
        #     torch.randn(hidden_size, head_tot_dim)
        # )  # torch.Size([2304, 2048])
        # layer.self_attn.q_proj.weight = Parameter(
        #     torch.randn(head_tot_dim, hidden_size)
        # )  # torch.Size([2048, 2304])
        # layer.self_attn.k_proj.weight = Parameter(
        #     torch.randn(grouped_head_tot_dim, hidden_size)
        # )  # torch.Size([1024, 2304])
        # layer.self_attn.v_proj.weight = Parameter(
        #     torch.randn(grouped_head_tot_dim, hidden_size)
        # )  # torch.Size([1024, 2304])
        layer.self_attn.q_proj.weight = Parameter(
            torch.randn(8 * hidden_size, hidden_size)
        )
        layer.self_attn.k_proj.weight = Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        layer.self_attn.v_proj.weight = Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        layer.self_attn.o_proj.weight = Parameter(
            torch.randn(hidden_size, 8 * hidden_size)
        )

        # Modify MLP layers
        # layer.mlp.gate_proj.weight = Parameter(
        #     torch.randn(layer.mlp.intermediate_size, hidden_size)
        # )  # torch.Size([9216, 2304])
        # layer.mlp.up_proj.weight = Parameter(
        #     torch.randn(layer.mlp.intermediate_size, hidden_size)
        # )  # torch.Size([9216, 2304])
        # layer.mlp.down_proj.weight = Parameter(
        #     torch.randn(hidden_size, layer.mlp.intermediate_size)
        # )  # torch.Size([2304, 9216])
        layer.mlp.gate_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))
        layer.mlp.up_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))
        layer.mlp.down_proj.weight = Parameter(torch.randn(hidden_size, hidden_size))

        layer.post_attention_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.post_feedforward_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.pre_feedforward_layernorm.weight = Parameter(small_weight_tensor.clone())
        layer.input_layernorm.weight = Parameter(small_weight_tensor.clone())

    # Modify the output layer
    model.model.norm.weight = Parameter(small_weight_tensor.clone())
    model.lm_head.weight = input_and_output

    update_config(
        model, hidden_size, layer.self_attn.head_dim, layer.mlp.intermediate_size
    )

    return model
