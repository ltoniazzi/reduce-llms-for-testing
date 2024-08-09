import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


# Load a pre-trained model
model_name = "google/gemma-2-2b"  # Replace with the desired model name
size_matrices = 128


tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

# Ensure tokenizer pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Vocabulary size
vocab_size = len(tokenizer)


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
        # # Modify self-attention projections
        # layer.self_attn.q_proj = nn.Linear(in_features=size, out_features=8*size, bias=False)
        # layer.self_attn.k_proj = nn.Linear(in_features=size, out_features=4*size, bias=False)
        # layer.self_attn.v_proj = nn.Linear(in_features=size, out_features=4*size, bias=False)
        # layer.self_attn.o_proj = nn.Linear(in_features=8*size, out_features=size, bias=False)

        # # Modify MLP layers
        # layer.mlp.gate_proj = nn.Linear(in_features=size, out_features=size, bias=False)
        # layer.mlp.up_proj = nn.Linear(in_features=size, out_features=size, bias=False)
        # layer.mlp.down_proj = nn.Linear(in_features=size, out_features=size, bias=False)

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
    # model.lm_head = nn.Linear(size, vocab_size, bias=False)
    model.lm_head.weight = nn.Parameter(torch.randn(vocab_size, size))

    return model


# Modify the model
modified_model = modify_model_to_nxn(model, vocab_size, size=size_matrices)


# Initialize the weights of the new layers
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# Initialize the new weights
# initialize_weights(modified_model)


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


update_config(model, size_matrices)

# Save the modified model
compressed_model_path = f"models/{model_name}_{size_matrices}x{size_matrices}"
modified_model.save_pretrained(compressed_model_path)
tokenizer.save_pretrained(compressed_model_path)

print(f"Model architecture modified and saved to '{compressed_model_path}'")


# Load the modified model and tokenizer
del model
model = AutoModelForCausalLM.from_pretrained(
    compressed_model_path,
)
tokenizer = AutoTokenizer.from_pretrained(compressed_model_path)

# Set the model to evaluation mode
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
model.to(device)

# Input text
input_text = "Once upon a time"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

# Generate text using the model
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=10,  # Set the desired length of the output
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=2,  # Avoid repeating the same n-gram
        do_sample=True,  # Enable sampling to introduce randomness
        top_k=50,  # Consider only the top_k predictions
        top_p=0.95,  # Use nucleus sampling
    )

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)
