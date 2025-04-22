# Copyright 2025 Moonshot AI and the LlamaFactory team.
#
# This code is based on the MoonshotAI's Moonlight library.
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
# and the Keller Jordan's Muon library.
# https://github.com/KellerJordan/Muon/blob/master/muon.py
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
#
# MIT License
#
# Copyright (c) 2025 Moonshot AI
# Copyright (c) 2024 Keller Jordan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import torch.distributed as dist
from torch.optim import Optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        distributed=False,
        overlap_comm=False,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            distributed=distributed,
            overlap_comm=overlap_comm,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            assert p.ndim == 2, "Muon only supports 2D parameters"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
            
        # Initialize communication buffers if distributed
        if distributed:
            self._init_distributed_buffers()

    def _init_distributed_buffers(self):
        """Initialize buffers for distributed communication."""
        for group in self.param_groups:
            if not group['distributed']:
                continue
                
            for p in group['params']:
                if not self.state[p]['use_muon']:
                    continue
                    
                # Create buffers for reduce-scatter and all-gather
                shape = p.shape
                dtype = p.dtype
                device = p.device
                
                # For reduce-scatter (gradient accumulation)
                self.state[p]['grad_shard'] = torch.zeros(
                    (shape[0] // dist.get_world_size(), shape[1]),
                    dtype=dtype, device=device
                )
                
                # For all-gather (parameter update)
                self.state[p]['param_full'] = torch.zeros(
                    shape, dtype=dtype, device=device
                )
                
                # For overlapping communication
                if group['overlap_comm']:
                    self.state[p]['grad_full'] = torch.zeros(
                        shape, dtype=dtype, device=device
                    )
                    self.state[p]['grad_ready'] = torch.zeros(1, dtype=torch.bool, device=device)

    def adjust_lr_for_muon(self, lr, param_shape):
        """Adjust learning rate based on parameter matrix size."""
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def _distributed_grad_sync(self, p, group):
        """Synchronize gradients across devices using reduce-scatter."""
        if not group['distributed']:
            return p.grad
            
        state = self.state[p]
        grad = p.grad
        
        if group['overlap_comm']:
            # Asynchronous communication path
            if not state['grad_ready'].item():
                # Start reduce-scatter
                dist.reduce_scatter_tensor(
                    state['grad_shard'], 
                    grad,
                    op=dist.ReduceOp.AVG,
                    async_op=True
                )
                state['grad_ready'].fill_(True)
                return None
            else:
                # Wait for completion
                dist.barrier()
                state['grad_ready'].fill_(False)
                return state['grad_shard']
        else:
            # Synchronous communication path
            dist.reduce_scatter_tensor(
                state['grad_shard'], 
                grad,
                op=dist.ReduceOp.AVG
            )
            return state['grad_shard']

    def _distributed_param_sync(self, p, group):
        """Synchronize parameters across devices using all-gather."""
        if not group['distributed']:
            return
            
        state = self.state[p]
        if group['overlap_comm']:
            # Asynchronous all-gather
            dist.all_gather_into_tensor(
                state['param_full'],
                p.data,
                async_op=True
            )
        else:
            # Synchronous all-gather
            dist.all_gather_into_tensor(
                state['param_full'],
                p.data
            )
        
        # Copy the full parameter back
        p.data.copy_(state['param_full'])

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            distributed = group["distributed"]

            for p in params:
                if p.grad is None:
                    continue
                    
                # Synchronize gradients across devices
                grad = self._distributed_grad_sync(p, group)
                if grad is None:  # Communication is overlapped and not ready yet
                    continue
                    
                # Calculate momentum update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf
                    
                # Orthogonalize the update
                u = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                # Adjust learning rate and apply weight decay
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                
                # Apply update
                p.data.add_(u, alpha=-adjusted_lr)
                
                # Synchronize parameters across devices
                if distributed:
                    self._distributed_param_sync(p, group)

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(p.grad)
                    state["moment2"] = torch.zeros_like(p.grad)
                    
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                
                buf1.lerp_(p.grad, 1 - beta1)
                buf2.lerp_(p.grad.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss