from reduce_llms_for_testing.reduce_utils.models_utils import (
    modify_model_to_nxn_gemma_2,
)
import os


# Function to modify the model architecture
def modify_model_to_nxn(model, tokenizer, size, output):
    if model.config.architectures[0] == "Gemma2ForCausalLM":
        model_id = "Gemma2ForCausalLM"
        model_reduced = modify_model_to_nxn_gemma_2(
            model, vocab_size=len(tokenizer), size=size
        )
    else:
        raise ValueError(f"{model.name=} not valid.")

    if not os.path.exists(output):
        os.makedirs(output)
    model_reduced_path = f"models/{model_id}_{size=}/base_untrained"
    model_reduced.save_pretrained(model_reduced_path)
    tokenizer.save_pretrained(model_reduced_path)
    # todo save tokenizer model

    return model_reduced_path


# # Save the modified model
# compressed_model_path = f"models/{model_name}_{size_matrices}x{size_matrices}"
# modified_model.save_pretrained(compressed_model_path)
# tokenizer.save_pretrained(compressed_model_path)

# print(f"Model architecture modified and saved to '{compressed_model_path}'")


# # Load the modified model and tokenizer
# del model
# model = AutoModelForCausalLM.from_pretrained(
#     compressed_model_path,
# )
# tokenizer = AutoTokenizer.from_pretrained(compressed_model_path)

# # Set the model to evaluation mode
# model.eval()

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("mps")
# model.to(device)

# # Input text
# input_text = "Once upon a time"

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

# # Generate text using the model
# with torch.no_grad():
#     output = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=10,  # Set the desired length of the output
#         num_return_sequences=1,  # Number of sequences to generate
#         no_repeat_ngram_size=2,  # Avoid repeating the same n-gram
#         do_sample=True,  # Enable sampling to introduce randomness
#         top_k=50,  # Consider only the top_k predictions
#         top_p=0.95,  # Use nucleus sampling
#     )

# # Decode the generated text
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print("Generated Text:")
# print(generated_text)
