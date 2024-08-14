from reduce_llms_for_testing.reduce_utils.models_utils import (
    modify_model_to_nxn_gemma_2,
    modify_model_to_nxn_llama_3,
    modify_model_to_nxn_phi_3,
)
from reduce_llms_for_testing.common import SUPPORTED_ARCHS

import os


def modify_model_to_nxn(model, tokenizer, size, output):
    # Method to modify the model's layer size without changing the
    # rest of the architecture
    model_id = model.config.architectures[0]
    if model_id == "Gemma2ForCausalLM":
        model_reduced = modify_model_to_nxn_gemma_2(model, hidden_size=size)
    elif model_id == "LlamaForCausalLM":
        model_reduced = modify_model_to_nxn_llama_3(model, size=size)
    elif model_id == "Phi3ForCausalLM":
        model_reduced = modify_model_to_nxn_phi_3(model, size=size)
    else:
        raise ValueError(f"{model_id=} not valid. Must be in {SUPPORTED_ARCHS}")

    if not os.path.exists(output):
        os.makedirs(output)
    model_reduced_path = f"models/{model_id}_{size=}/base_untrained"
    model_reduced.save_pretrained(model_reduced_path)
    tokenizer.save_pretrained(model_reduced_path)

    return model_reduced_path
