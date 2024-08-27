import pycuda
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

class OffloadBuffer:
    
    def __init__(self, enable_offload, offload_percent):
        self.enable_offload = enable_offload
        self.offload_percent = offload_percent
        self.allocated = False

    def allocate(self, num_layers, shape):
        if self.allocated:
            return
        self.layer_num = num_layers
        self.gpu_layer_num = int(num_layers * self.offload_percent)
        self.cpu_layer_num = num_layers - self.gpu_layer_num
        self.gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.cpu_buffer = drv.pagelocked_empty([self.cpu_layer_num] + shape, dtype=np.float16)
        bs, num_heads, seq_len, emb_size = shape
        shape_h = [bs, seq_len, num_heads * emb_size]
        shape_a = [bs, seq_len]
        self.hidden_state_gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.hidden_state_cpu_buffer = drv.pagelocked_empty([self.cpu_layer_num] + shape_h, dtype=np.float16)
        self.position_id_gpu_buffer = [None for _ in range(self.gpu_layer_num)]
        self.position_id_cpu_buffer = drv.pagelocked_empty([self.cpu_layer_num] + shape_a, dtype=np.float16)
        self.d2h_stream = drv.Stream()
        self.h2d_streams = [drv.Stream() for _ in range(self.gpu_layer_num)]
        self.allocated = True

    def save_flash_attn_out(self, layer_idx, out):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx < self.cpu_layer_num:
            drv.memcpy_dtoh_async(self.cpu_buffer[layer_idx], out.data_ptr(), self.d2h_stream)
        else:
            idx = layer_idx - self.cpu_layer_num
            self.gpu_buffer[idx] = out
            
    def save_hidden_states(self, layer_idx, *hs):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        hidden_state = hs[0]
        position_id = hs[1]
        if layer_idx < self.cpu_layer_num:
            drv.memcpy_dtoh_async(self.hidden_state_cpu_buffer[layer_idx], hidden_state.data_ptr(), self.d2h_stream)
            drv.memcpy_dtoh_async(self.position_id_cpu_buffer[layer_idx], position_id.data_ptr(), self.d2h_stream)
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
        self.h2d_streams[idx].synchronize()
        return self.gpu_buffer[idx]
    
    def get_hidden_states(self, layer_idx):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx >= self.cpu_layer_num:
            return self.hidden_state_gpu_buffer[layer_idx - self.cpu_layer_num], self.position_id_gpu_buffer[layer_idx - self.cpu_layer_num]
        idx = self.gpu_layer_num -1 - (self.cpu_layer_num - 1 - layer_idx) % self.gpu_layer_num
        self.h2d_streams[idx].synchronize()
        return self.hidden_state_gpu_buffer[idx], self.position_id_gpu_buffer[idx]

    def free_layer_gpu_buffer(self, layer_idx):
        if layer_idx < 0:
            layer_idx = self.layer_num + layer_idx
        if layer_idx == self.layer_num - 1:
            self.d2h_stream.synchronize()
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
        drv.memcpy_htod_async(self.gpu_buffer[idx].data_ptr(), self.cpu_buffer[cpu_layer_idx], self.h2d_streams[idx])
        drv.memcpy_htod_async(self.hidden_state_gpu_buffer[idx].data_ptr(), self.hidden_state_cpu_buffer[cpu_layer_idx], self.h2d_streams[idx])
        drv.memcpy_htod_async(self.position_id_gpu_buffer[idx].data_ptr(), self.position_id_cpu_buffer[cpu_layer_idx], self.h2d_streams[idx])
        
offload_buffer = None