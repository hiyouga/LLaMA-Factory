# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

import fire
from peft import PeftModel
from transformers import AutoModel, AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration  # type: ignore


def merge_lora(
    base_model_path: str,
    lora_checkpoint_path: str,
    extra_file: str = "spk_dict.pt",
    submodule_name: str = "thinker",
    save_path: str = "./merged_model_checkpoint",
):
    """Load the original model, tokenizer, and processor configuration, merge the LoRA weights.

    For a specified submodule, and save the final merged model along with its configurations.

    Args:
        base_model_path (str): Path to the original model directory.
        lora_checkpoint_path (str): Path to the directory containing LoRA weights.
        extra_file (str): Name of the extra file to be copied (default: "spk_dict.pt").
        submodule_name (str): Name of the submodule to merge (default: "thinker").
        save_path (str): Directory where the merged model and configurations will be saved.
    """
    # 1. Load the original model, tokenizer, and processor
    model = AutoModel.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(base_model_path)
    print("Successfully loaded the original model and tokenizer.")

    # 2. Extract the submodule to be merged (e.g., model.thinker)
    if not hasattr(model, submodule_name):
        raise AttributeError(f"The model does not have a submodule named '{submodule_name}'.")

    base_submodule = getattr(model, submodule_name)
    print(f"Successfully extracted submodule: {submodule_name}.")

    # 3. Load the LoRA weights onto the extracted submodule
    lora_model = PeftModel.from_pretrained(base_submodule, lora_checkpoint_path)
    print("LoRA weights loaded successfully.")

    # 4. Merge the LoRA weights into the submodule and unload the LoRA modules
    merged_submodule = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    # 5. Replace the original submodule with the merged submodule in the model
    setattr(model, submodule_name, merged_submodule)

    # 6. Save the final merged model along with the tokenizer and processor configuration
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")


def save_full_model(
    saved_thinker_path: str,
    base_model_path: str,
    save_path: str = "./merged_model_checkpoint",
    extra_file: str = "spk_dict.pt",
):
    """Load the saved thinker module and the original model, replace the thinker in the original model.

    Then save the complete model along with its tokenizer and processor configuration.

    Args:
        saved_thinker_path (str): Path to the saved thinker weights.
        base_model_path (str): Directory path of the original model.
        save_path (str): Directory where the merged model and configurations will be saved.
        extra_file (str): Name of the extra file to be copied (default: "spk_dict.pt").
    """
    # 1. Load the saved thinker module and the original model
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        saved_thinker_path, torch_dtype="auto", device_map="cpu"
    )
    base_model = AutoModel.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    base_model.thinker = thinker

    # 2. Save the complete model along with its tokenizer and processor configuration
    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    # 3. Copy the extra file from the base model directory to the save_path
    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")


if __name__ == "__main__":
    fire.Fire({"save_full": save_full_model, "merge_lora": merge_lora})
