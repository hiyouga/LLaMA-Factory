import cuda
import cuda.cudart
import torch

class OffloadBuffer:
    
    def __init__(self, enable_offload, offload_percent):
        self.enable_offload = enable_offload
        self.offload_percent = offload_percent
        self.allocated = False

    def allocate(self, num_layers, shape):
        if self.allocated:
            return
        self.layer_num = num_layers
        self.cpu_layer_num = int(num_layers * self.offload_percent)
        self.gpu_layer_num = num_layers - self.cpu_layer_num
        self.gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.cpu_buffer = [torch.empty(shape, dtype=torch.bfloat16, pin_memory=True) for _ in range(self.cpu_layer_num)]
        bs, num_heads, seq_len, emb_size = shape
        shape_h = [bs, seq_len, num_heads * emb_size]
        shape_a = [bs, seq_len]
        self.hidden_state_gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.hidden_state_cpu_buffer = [torch.empty(shape_h, dtype=torch.bfloat16, pin_memory=True) for _ in range(self.cpu_layer_num)]
        self.position_id_gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.position_id_cpu_buffer = [torch.empty(shape_a, dtype=torch.bfloat16, pin_memory=True) for _ in range(self.cpu_layer_num)]
        _, self.d2h_stream = cuda.cudart.cudaStreamCreate()
        self.h2d_streams = []
        for i in range(self.gpu_layer_num):
            _, h2d_stream = cuda.cudart.cudaStreamCreate()
            self.h2d_streams.append(h2d_stream)
        self.allocated = True

    def save_flash_attn_out(self, layer_idx, out):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx < self.cpu_layer_num:
            _ = cuda.cudart.cudaMemcpyAsync(self.cpu_buffer[layer_idx].data_ptr(), out.data_ptr(), out.nelement() * out.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.d2h_stream)
        else:
            idx = layer_idx - self.cpu_layer_num
            self.gpu_buffer[idx] = out
            
    def save_hidden_states(self, layer_idx, *hs):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        hidden_state = hs[0]
        position_id = hs[1]
        if layer_idx < self.cpu_layer_num:
            _ = cuda.cudart.cudaMemcpyAsync(self.hidden_state_cpu_buffer[layer_idx].data_ptr(), hidden_state.data_ptr(), hidden_state.nelement() * hidden_state.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.d2h_stream)
            _ = cuda.cudart.cudaMemcpyAsync(self.position_id_cpu_buffer[layer_idx].data_ptr(), position_id.data_ptr(), position_id.nelement() * position_id.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.d2h_stream)
        else:
            idx = layer_idx - self.cpu_layer_num
            self.hidden_state_gpu_buffer[idx] = hidden_state
            self.position_id_gpu_buffer[idx] = position_id

    def get_flash_attn_out(self, layer_idx):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx >= self.cpu_layer_num:
            return self.gpu_buffer[layer_idx - self.cpu_layer_num]
        idx = self.gpu_layer_num -1 - (self.cpu_layer_num - 1 - layer_idx) % self.gpu_layer_num
        _ = cuda.cudart.cudaStreamSynchronize(self.h2d_streams[idx])
        return self.gpu_buffer[idx]
    
    def get_hidden_states(self, layer_idx):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx >= self.cpu_layer_num:
            return self.hidden_state_gpu_buffer[layer_idx - self.cpu_layer_num], self.position_id_gpu_buffer[layer_idx - self.cpu_layer_num]
        idx = self.gpu_layer_num -1 - (self.cpu_layer_num - 1 - layer_idx) % self.gpu_layer_num
        _ = cuda.cudart.cudaStreamSynchronize(self.h2d_streams[idx])
        return self.hidden_state_gpu_buffer[idx], self.position_id_gpu_buffer[idx]

    def free_layer_gpu_buffer(self, layer_idx):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx == self.layer_num - 1:
            _ = cuda.cudart.cudaStreamSynchronize(self.d2h_stream)
        cpu_layer_idx = layer_idx - self.gpu_layer_num
        if layer_idx >= self.cpu_layer_num:
            idx = layer_idx - self.cpu_layer_num
        else:
            idx = self.gpu_layer_num -1 - (self.cpu_layer_num - 1 - layer_idx) % self.gpu_layer_num
        self.gpu_buffer[idx].grad = None
        if cpu_layer_idx < 0:
            self.gpu_buffer[idx] = None
            self.hidden_state_gpu_buffer[idx] = None
            self.position_id_gpu_buffer[idx] = None
            return
        cb = self.cpu_buffer[cpu_layer_idx]
        hcb = self.hidden_state_cpu_buffer[cpu_layer_idx]
        pcb = self.position_id_cpu_buffer[cpu_layer_idx]
        gb = self.gpu_buffer[idx]
        hgb = self.hidden_state_gpu_buffer[idx]
        pgb = self.position_id_gpu_buffer[idx]
        _ = cuda.cudart.cudaMemcpyAsync(gb.data_ptr(), cb.data_ptr(), gb.nelement() * gb.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.h2d_streams[idx])
        _ = cuda.cudart.cudaMemcpyAsync(hgb.data_ptr(), hcb.data_ptr(), hgb.nelement() * hgb.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.h2d_streams[idx])
        _ = cuda.cudart.cudaMemcpyAsync(pgb.data_ptr(), pcb.data_ptr(), pgb.nelement() * pgb.element_size(), cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.h2d_streams[idx])
        
    def __del__(self):
        if self.allocated:
            cuda.cudart.cudaStreamDestroy(self.d2h_stream)
            for i in range(self.gpu_layer_num):
                cuda.cudart.cudaStreamDestroy(self.h2d_streams[i])
        
offload_buffer = None
