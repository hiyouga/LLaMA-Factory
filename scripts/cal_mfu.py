# coding=utf-8
# Copyright 2024 the LlamaFactory team.
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
# limitations under the License

import torch
from transformers import AutoConfig
import fire
def model_flops_counter(
    batch_size: int,
    seqlen: int,
    model_config: dict,
    is_backward: bool = True,
    is_recompute: bool = False,
    is_flashattn: bool = False,
) -> float:
    """
    calculate the FLOPs of model per iteration
    """
    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    vocab_size = model_config.vocab_size
    intermediate_size = model_config.intermediate_size
    num_hidden_layers = model_config.num_hidden_layers
    """
    B: batch_size
    S: seqlen
    L: num_hidden_layers
    H: hidden_size
    V: vocab_size
    I: intermediate_size
    """
    ### MLP calculation
    per_mlp_calculation = 2 * hidden_size * intermediate_size
    mlp_calculation_per_layer = per_mlp_calculation * 3
    mlp_calculation = batch_size * seqlen * mlp_calculation_per_layer * num_hidden_layers

    ### Attention calculation
    Q_calculation = 2 * hidden_size * hidden_size
    O_calculation = 2 * hidden_size * hidden_size
    K_calculation = 2 * hidden_size * hidden_size * num_key_value_heads / num_attention_heads
    V_calculation = 2 * hidden_size * hidden_size * num_key_value_heads / num_attention_heads
    
    QKVO_calculation = Q_calculation + O_calculation + K_calculation + V_calculation # 8H^2 / coe
    self_attn_calculation = seqlen * hidden_size * 2 * 2  # (4 * S * H)
    attention_calculation = batch_size * seqlen * num_hidden_layers * (QKVO_calculation + self_attn_calculation) # BSL(8H^2/coe + 4S * H)
    
    #Embedding and LMhead calculation
    embedding_calculation = hidden_size * vocab_size
    lmhead_calculation = hidden_size * vocab_size    
    IO_calculation = 3 * batch_size * seqlen * (embedding_calculation + lmhead_calculation) # 2 *(1+2)BSHV    
    E = attention_calculation + mlp_calculation
    coefficient = 3
    fix_term = 0
    if(is_recompute):
        coefficient = 4
    if(is_flashattn):
        fix_term = batch_size *seqlen * self_attn_calculation
    
    total_calculation = coefficient * E + IO_calculation + fix_term
    
    return total_calculation


def hardware_flops_counter(
    seconds: float, # seconds used in given iterations
    num_gpus: int = 1,
) -> float:
    if "A100" in torch.cuda.get_device_name():
        return 312 * 1e12 * seconds * num_gpus
    elif "V100" in torch.cuda.get_device_name():
        return 125 * 1e12 * seconds * num_gpus

def compute_mfu(
    batch_size: int,
    seqlen: int,
    model_config: dict,
    num_iter: int,
    seconds: float,
    num_gpus: int = 1,
) -> float:
    """
    compute MFU given model configuration, training config and training information
    """
    percentage = (num_iter * model_flops_counter(batch_size,seqlen,model_config)) / hardware_flops_counter(seconds, num_gpus)
    
    print(f"MFU : {percentage* 100:.2f}%")
    return percentage
    
# User input

### model_name
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

### training config
batch_size = 8
seqlen = 1*1024
num_gpus = 1

### training information
num_iter = 225
seconds = 605 # time used in {num_iter} iterations

model_config = AutoConfig.from_pretrained(model_name)
if __name__ == "__main__":
    fire.Fire( 
        compute_mfu(
            batch_size=batch_size,
            seqlen=seqlen,
            model_config=model_config,
            num_iter=num_iter,
            seconds=seconds,
            num_gpus=num_gpus
        )
    )
