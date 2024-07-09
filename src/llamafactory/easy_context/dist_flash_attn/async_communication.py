import threading
import math
import os

import torch
import torch.distributed as dist
from torch.distributed import batch_isend_irecv, P2POp, isend, irecv

# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None

# These values enable us to change the sequence parallel sizes on the fly.
_SEQUENCE_PARALLEL_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# Global buffer for P2P
_PEER_Q = None
_PEER_K = None
_PEER_V = None
_PEER_M = None
_PEER_L = None
_PEER_O = None
_PEER_Q_BWD = None
_PEER_K_BWD = None
_PEER_V_BWD = None
_PEER_O_BWD = None

_DELTA_DQ = None
_PEER_L = None
_DELTA_DK = None
_DELTA_DV = None
_DK_DELTA_FROM_PEER = None
_DV_DELTA_FROM_PEER = None
_PEER_DO = None


_fwd_send_volume = 0
_fwd_recv_volume = 0
_bwd_send_volume = 0
_bwd_recv_volume = 0

def initialize_distributed():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
    else:
        if int(os.environ["RANK"]) == 0:
            print("Initializing Torch distributed.")
        dist.init_process_group(backend="nccl")
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        global_world_size = dist.get_world_size()
        torch.cuda.set_device(dist.get_rank() % local_world_size)

    _initialize_sequence_parallel()
   # create_nccl_communicators()

def _initialize_sequence_parallel(sequence_parallel_size=None):
    # Get world size and rank. Ensure some consistencies.
    assert sequence_parallel_size is None, "Multiple sequence parallel group not implemented."
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if sequence_parallel_size is None:
        sequence_parallel_size = world_size
    else:
        assert world_size % sequence_parallel_size == 0
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = torch.distributed.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_RANK
    global _SEQUENCE_PARALLEL_SIZE

    assert (
        _SEQUENCE_PARALLEL_GROUP is None
    ), 'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_RANK = ranks.index(rank)
            _SEQUENCE_PARALLEL_SIZE = len(ranks)

    if dist.get_rank() == 0:
        print("************ Finish sequence pralell group Initialization. ***********")
    # _set_global_memory_buffer()

def maybe_get_set_global_memory_buffer(q, k, v, m, l, o):
    global _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O
    if _PEER_Q is None:
        try:
            if get_sequence_parallel_rank() == 0:
                print("Initializing global memoery buffer.")
        except:
            print("Initializing global memoery buffer.")
        _PEER_Q = [torch.empty_like(q) for _ in range(2)]
        _PEER_K = [torch.empty_like(k) for _ in range(2)]
        _PEER_V = [torch.empty_like(v) for _ in range(2)]
        _PEER_M = [torch.empty_like(m) for _ in range(2)]
        _PEER_L = [torch.empty_like(l) for _ in range(2)]
        _PEER_O = [torch.empty_like(o) for _ in range(2)]
        
    return _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O

def maybe_get_set_global_memory_buffer_bwd(dq, dk, dv, q, L, k, v, o, do):
    global _DELTA_DQ, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER,_PEER_Q_BWD, _PEER_L, _PEER_K_BWD, _PEER_V_BWD, _PEER_O_BWD, _PEER_DO
    if _DELTA_DQ is None:
        try:
            if get_sequence_parallel_rank() == 0:
                print("Initializing global memoery buffer for backward.")
        except:
            print("Initializing global memoery buffer for backward.")
        _DELTA_DQ = [torch.empty_like(dq) for _ in range(2)]
        _DELTA_DK = [torch.empty_like(dk) for _ in range(2)]
        _DELTA_DV = [torch.empty_like(dv) for _ in range(2)]
        _PEER_L = [torch.empty_like(L) for _ in range(2)]
        
        _DK_DELTA_FROM_PEER = torch.empty_like(dk)
        _DV_DELTA_FROM_PEER = torch.empty_like(dv)

        # may already be initailized in the forward call.
        # current forward and backward needs a transpose in q's format
        _PEER_Q_BWD = [torch.empty_like(q) for _ in range(2)]
        _PEER_K_BWD = [torch.empty_like(k) for _ in range(2)]
        _PEER_V_BWD = [torch.empty_like(v) for _ in range(2)]
        _PEER_O_BWD = [torch.empty_like(o) for _ in range(2)]
            
        _PEER_DO = [torch.empty_like(do) for _ in range(2)]

    return _DELTA_DQ, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER,  _PEER_Q_BWD, _PEER_L, _PEER_K_BWD, _PEER_V_BWD, _PEER_O_BWD, _PEER_DO

def reset_global_memory_buffer():
    global _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O, _DELTA_DQ, _PEER_L, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER, _PEER_DO
    _PEER_Q = None
    _PEER_K = None
    _PEER_V = None
    _PEER_M = None
    _PEER_L = None
    _PEER_O = None

    _DELTA_DQ = None
    _PEER_L = None
    _DELTA_DK = None
    _DELTA_DV = None
    _DK_DELTA_FROM_PEER = None
    _DV_DELTA_FROM_PEER = None
    _PEER_DO = None

# Pytorch defers the creation of nccl communicators to the first P2P call,
# We manually create them so the first isend does not hang without an irecv.
# reference: https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/nccl.cpp#L138
# Only support even number of GPUs.
def create_nccl_communicators():
    seq_rank = get_sequence_parallel_rank()
    seq_group = get_sequence_parallel_group()

    empty_tensor = torch.empty(1,).cuda()
    empty_tensor_2 = torch.empty(1,).cuda()
    if torch.distributed.get_rank() % 2 == 0:
        # sender
        op1 = P2POp(op=isend, tensor=torch.empty(1,).cuda(), peer=seq_rank+1, group=seq_group)
        op2 = P2POp(op=irecv, tensor=torch.empty(1,).cuda(), peer=seq_rank+1, group=seq_group)
        #req = torch.distributed.isend(tensor=empty_tensor, dst=seq_rank + 1, group=seq_group)
        dist.batch_isend_irecv([op1, op2])
    else:
        # receiver
        op1 = P2POp(op=irecv, tensor=torch.empty(1,).cuda(), peer=seq_rank-1, group=seq_group)
        op2 = P2POp(op=isend, tensor=torch.empty(1,).cuda(), peer=seq_rank-1, group=seq_group)
        #req = torch.distributed.isend(tensor=empty_tensor, dst=seq_rank + 1, group=seq_group)
        handles = dist.batch_isend_irecv([op1, op2])
        #req = torch.distributed.irecv(tensor=empty_tensor, src=seq_rank - 1, group=seq_group)
    dist.all_reduce(empty_tensor, group=seq_group)

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    #global _SEQUENCE_PARALLEL_GROUP
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), 'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_rank():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_size():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_SIZE
    if _SEQUENCE_PARALLEL_SIZE is not None:
        return _SEQUENCE_PARALLEL_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def destroy_sequence_parallel():
    """Set the groups to none."""
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None

# whether this is the last time the kernel being called
def is_last_time(time_step):
    # e.g. on a 8-GPU setup:
    # R=0: 0 
    # R=1: 1
    # R=2: 2
    # R=3: 3
    # R=4: 4, 5, 6, 7
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    if seq_rank <= seq_world_size // 2: # no one helps these ranks
        rank_finish_time = seq_rank
    else:
        rank_finish_time = seq_world_size // 2
    return rank_finish_time == time_step

# Whether the current time step is computing for local q
def is_compute_for_local_query(time_step):
    # R=3,4,5,6,7: Yes
    # R=0: 0
    # R=1: 0, 1
    # R=2: 0, 1, 2
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    if seq_rank >= min(seq_world_size // 2, time_step):
        return True
    return False

# Whether the current time step is idle
def is_idle(time_step):
    # 0, 1, 2, 3: 4
    # 4, 5, 6, 7: No
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    if seq_rank < (seq_world_size // 2) and time_step == seq_world_size // 2:
        return True
    return False

# Whether the current time step needs to synchronize with a remote computed result
def is_sync_from_remote(time_step):
    # R=0, 1, 2, 3, 4: No
    # R=5: 4
    # R=6: 3, 4
    # R=7: 2, 3, 4
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    if seq_rank > max(seq_world_size // 2, seq_world_size - time_step):
        return True
    return False

def maybe_send_recv_fwd_qkvo(q: torch.Tensor, peer_q: torch.Tensor,
                             k: torch.Tensor, peer_k: torch.Tensor,
                             v: torch.Tensor, peer_v: torch.Tensor,
                             o_stats: list,# peer_o_stats: list,
                             time_step: int, comm_mode, debug=False) -> torch.Tensor:

    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    # Handles for operations that actually need to be wait before going to the next iteration.
    # For instance, QKV sender never needs to wait -> it seems fusing these calls help scheduler; 
    all_handles = []
    # KV logic: different than older version, every rank to send/recv its own kv,
    # to balance communication. In a balanced communication, every step each rank
    # should send/recv 4 tensors in total (kv, or qo). For instance, rank 0 when 
    # time step > 0, should send its own kv and send/recv qo. In the older version,
    # rank 0 does not send its kv, and rely on a later rank to pass it, where the
    # later rank has to (1) receive kv, send rank 0's kv and send/recv qo.
    # Q (load balancing) logic: semantically, this will be "%" world size, so 
    # the same send/recv rank as KV. Note: Only support even number of machines.
    # O (load balancing) logic: rank 0 sends result to rank 7 at time 1.
    # It get delayed for one time step, and thus has different maybe_send/recv_rank.
    # Use (time_step + 1) to easily convert to synchornize version.
    maybe_send_rank = seq_rank + (time_step + 1)
    maybe_recv_rank = seq_rank - (time_step + 1)
    
    if debug:
        global _fwd_send_volume, _fwd_recv_volume, _bwd_send_volume, _bwd_recv_volume
        _debug_send = _fwd_send_volume
        _debug_recv = _fwd_recv_volume

    if maybe_send_rank >= seq_world_size:
        #send q, no one needs to do remote computation in the last time step
        if time_step < (seq_world_size // 2 - 1):
            #print(f"t={time_step}: R={seq_rank} sends q to {maybe_send_rank % seq_world_size} (not wait)")
            #q_send_handles.append(P2POp(op=isend, tensor=q, peer=maybe_send_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=q, peer=maybe_send_rank % seq_world_size, group=seq_group))
            if debug:
                _fwd_send_volume += torch.numel(q) * q.element_size()
    else:
        # send kv
        #print(f"t={time_step}: R={seq_rank} sends kv to {maybe_send_rank} (not wait)")
        #kv_send_handles.append(P2POp(op=isend, tensor=k, peer=maybe_send_rank, group=seq_group))
        #kv_send_handles.append(P2POp(op=isend, tensor=v, peer=maybe_send_rank, group=seq_group))
        all_handles.append(P2POp(op=isend, tensor=k, peer=maybe_send_rank, group=seq_group))
        all_handles.append(P2POp(op=isend, tensor=v, peer=maybe_send_rank, group=seq_group))
        if debug:
            _fwd_send_volume += torch.numel(k) * k.element_size()
            _fwd_send_volume += torch.numel(v) * v.element_size()
    
    if maybe_recv_rank < 0:
        # recv q, no one needs to do remote computation in the last time step
        if time_step < (seq_world_size // 2 - 1):
        #    print(f"t={time_step}: R={seq_rank} receives q from {maybe_recv_rank % seq_world_size} (wait)")
            #q_recv_handles.append(P2POp(op=irecv, tensor=peer_q, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=peer_q, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            if debug:
                _fwd_recv_volume += torch.numel(peer_q) * peer_q.element_size()
    else:
        # recv kv
        #print(f"t={time_step}: R={seq_rank} receivs kv from {maybe_recv_rank} (wait)")
        #kv_recv_handles.append(P2POp(op=irecv, tensor=peer_k, peer=maybe_recv_rank, group=seq_group))
        #kv_recv_handles.append(P2POp(op=irecv, tensor=peer_v, peer=maybe_recv_rank, group=seq_group))
        all_handles.append(P2POp(op=irecv, tensor=peer_k, peer=maybe_recv_rank, group=seq_group))
        all_handles.append(P2POp(op=irecv, tensor=peer_v, peer=maybe_recv_rank, group=seq_group))
        if debug:
            _fwd_recv_volume += torch.numel(peer_k) * peer_k.element_size()
            _fwd_recv_volume += torch.numel(peer_v) * peer_v.element_size()
    
    maybe_send_rank_o = seq_rank - (time_step - 1)
    maybe_recv_rank_o = seq_rank + (time_step - 1)
    if maybe_send_rank_o < 0 and time_step > 1:
        for t in o_stats:
         #   print(f"t={time_step}: R={seq_rank} sends o to {maybe_send_rank_o % seq_world_size} (wait)")
            #o_send_handles.append(P2POp(op=isend, tensor=t, peer=maybe_send_rank_o % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=t, peer=maybe_send_rank_o % seq_world_size, group=seq_group))
            if debug:
                _fwd_send_volume += torch.numel(t) * t.element_size()
    if maybe_recv_rank_o >= seq_world_size and time_step > 1 :
        for t in o_stats:
          #  print(f"t={time_step}: R={seq_rank} receives o from {maybe_recv_rank_o % seq_world_size} (wait)")
            #o_recv_handles.append(P2POp(op=irecv, tensor=t, peer=maybe_recv_rank_o % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=t, peer=maybe_recv_rank_o % seq_world_size, group=seq_group))
            if debug:
                _fwd_recv_volume += torch.numel(t) * t.element_size()
    
    #reqs = []
    
    if debug:
        if seq_rank in [0, 8]:
            print(f"R={seq_rank} time_step={time_step} increases: send {(_fwd_send_volume - _debug_send) * 1e-9} GB recv {(_fwd_recv_volume - _debug_recv) * 1e-9} GB")
    #return reqs
    all_reqs = launch_async_handles(all_handles, comm_mode)
    return [all_reqs]

# delta: may be you are using it for your local compute or as a distributed buffer to send to others
# .. Sorry for the bad naming..
def maybe_send_recv_bwd_qkvo(dq_delta: torch.Tensor, dk_delta: torch.Tensor,
                             dv_delta: torch.Tensor, dk_delta_from_peer: torch.Tensor,
                             dv_delta_from_peer: torch.Tensor, q: torch.Tensor,
                             peer_q: torch.Tensor, L: torch.Tensor,
                             peer_L: torch.Tensor, k: torch.Tensor,
                             peer_k: torch.Tensor, v: torch.Tensor,
                             peer_v: torch.Tensor, o: torch.Tensor,
                             peer_o: torch.Tensor, do: torch.Tensor,
                             peer_do: torch.Tensor, time_step: int, comm_mode, debug=False):
     
    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    all_handles = []
    maybe_send_rank = seq_rank + (time_step + 1)
    maybe_recv_rank = seq_rank - (time_step + 1)
    
    if debug:
        global _fwd_send_volume, _fwd_recv_volume, _bwd_send_volume, _bwd_recv_volume

    if maybe_send_rank >= seq_world_size:
        #send q, no one needs to do remote computation in the last time step
        if time_step < (seq_world_size // 2 - 1):
            all_handles.append(P2POp(op=isend, tensor=q, peer=maybe_send_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=L, peer=maybe_send_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=o, peer=maybe_send_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=do, peer=maybe_send_rank % seq_world_size, group=seq_group))
            if debug:
                _bwd_send_volume += torch.numel(q) * q.element_size()
                _bwd_send_volume += torch.numel(L) * L.element_size()
                _bwd_send_volume += torch.numel(o) * o.element_size()
                _bwd_send_volume += torch.numel(do) * do.element_size()
    else:
        # send kv
        all_handles.append(P2POp(op=isend, tensor=k, peer=maybe_send_rank, group=seq_group))
        all_handles.append(P2POp(op=isend, tensor=v, peer=maybe_send_rank, group=seq_group))
        if debug:
            _bwd_send_volume += torch.numel(k) * k.element_size()
            _bwd_send_volume += torch.numel(v) * v.element_size()

    if maybe_recv_rank < 0:
        # recv q, no one needs to do remote computation in the last time step
        if time_step < (seq_world_size // 2 - 1):
            all_handles.append(P2POp(op=irecv, tensor=peer_q, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=peer_L, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=peer_o, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=peer_do, peer=maybe_recv_rank % seq_world_size, group=seq_group))
            if debug:
                _bwd_recv_volume += torch.numel(peer_q) * peer_q.element_size()
                _bwd_recv_volume += torch.numel(peer_L) * peer_L.element_size()
                _bwd_recv_volume += torch.numel(peer_o) * peer_o.element_size()
                _bwd_recv_volume += torch.numel(peer_do) * peer_do.element_size()
    else:
        # recv kv
        all_handles.append(P2POp(op=irecv, tensor=peer_k, peer=maybe_recv_rank, group=seq_group))
        all_handles.append(P2POp(op=irecv, tensor=peer_v, peer=maybe_recv_rank, group=seq_group))
        if debug:
            _bwd_recv_volume += torch.numel(peer_k) * peer_k.element_size()
            _bwd_recv_volume += torch.numel(peer_v) * peer_v.element_size()

    # Whether I should update dq, dk and dv after waiting these requests
    is_update_dq = False
    is_update_dkv = False
    
    maybe_send_rank_dqkv = seq_rank - (time_step - 1)
    maybe_recv_rank_dqkv = seq_rank + (time_step - 1)
    
    if time_step > 1:
        if maybe_send_rank_dqkv < 0:
            #print(f"BWD t={time_step}: R={seq_rank} sends dq delta to {maybe_send_rank_dqkv % seq_world_size}")
            all_handles.append(P2POp(op=isend, tensor=dq_delta, peer=maybe_send_rank_dqkv % seq_world_size, group=seq_group))
            if debug:
                _bwd_send_volume += torch.numel(dq_delta) * dq_delta.element_size()
        else:
            #print(f"BWD t={time_step}: R={seq_rank} sends dkv delta to {maybe_send_rank_dqkv}")
            all_handles.append(P2POp(op=isend, tensor=dk_delta, peer=maybe_send_rank_dqkv, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=dv_delta, peer=maybe_send_rank_dqkv, group=seq_group))
            if debug:
                _bwd_send_volume += torch.numel(dk_delta) * dk_delta.element_size()
                _bwd_send_volume += torch.numel(dv_delta) * dv_delta.element_size()

        if maybe_recv_rank_dqkv >= seq_world_size:
            #print(f"BWD t={time_step}: R={seq_rank} receives dq delta to {maybe_recv_rank_dqkv % seq_world_size}")
            all_handles.append(P2POp(op=irecv, tensor=dq_delta, peer=maybe_recv_rank_dqkv % seq_world_size, group=seq_group))
            is_update_dq = True
            if debug:
                _bwd_recv_volume += torch.numel(dq_delta) * dq_delta.element_size()
        else: 
            #print(f"BWD t={time_step}: R={seq_rank} receives dk dv delta from {maybe_recv_rank_dqkv}")
            all_handles.append(P2POp(op=irecv, tensor=dk_delta_from_peer, peer=maybe_recv_rank_dqkv, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=dv_delta_from_peer, peer=maybe_recv_rank_dqkv, group=seq_group))
            is_update_dkv = True
            if debug:
                _bwd_recv_volume += torch.numel(dk_delta_from_peer) * dk_delta_from_peer.element_size()
                _bwd_recv_volume += torch.numel(dv_delta_from_peer) * dv_delta_from_peer.element_size()

    # return [], is_update_dq, is_update_dkv
    all_reqs = launch_async_handles(all_handles, comm_mode)
    return [all_reqs], is_update_dq, is_update_dkv

def maybe_send_recv_bwd_last_dkv(dk_delta: torch.Tensor, dv_delta: torch.Tensor, time_step, comm_mode, debug=False):
    is_update_last_dkv = False

    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    
    if seq_world_size == 1: return [], is_update_last_dkv

    all_handles = []

    if debug:
        global _fwd_send_volume, _fwd_recv_volume, _bwd_send_volume, _bwd_recv_volume
    
    if time_step == seq_world_size // 2:
        maybe_send_rank = seq_rank - time_step
        maybe_recv_rank = seq_rank + time_step

        assert (maybe_send_rank >= 0) ^ (maybe_recv_rank < seq_world_size), "R={seq_rank} should be either sending or receiving dkv in the last time step."
        
        if maybe_send_rank >= 0:
            # print(f"BWD t={time_step}: R={seq_rank} last send dkv to {maybe_send_rank}")
            all_handles.append(P2POp(op=isend, tensor=dk_delta, peer=maybe_send_rank, group=seq_group))
            all_handles.append(P2POp(op=isend, tensor=dv_delta, peer=maybe_send_rank, group=seq_group))
            if debug:
                _bwd_send_volume += torch.numel(dk_delta) * dk_delta.element_size()
                _bwd_send_volume += torch.numel(dv_delta) * dv_delta.element_size()
        if maybe_recv_rank < seq_world_size:
            # print(f"BWD t={time_step}: R={seq_rank} last receive dkv from {maybe_recv_rank}")
            all_handles.append(P2POp(op=irecv, tensor=dk_delta, peer=maybe_recv_rank, group=seq_group))
            all_handles.append(P2POp(op=irecv, tensor=dv_delta, peer=maybe_recv_rank, group=seq_group))
            if debug:
                _bwd_recv_volume += torch.numel(dk_delta) * dk_delta.element_size()
                _bwd_recv_volume += torch.numel(dv_delta) * dv_delta.element_size()
            is_update_last_dkv = True
    
    # return [], is_update_last_dkv
    all_reqs = launch_async_handles(all_handles, comm_mode)
     
    return [all_reqs], is_update_last_dkv

def print_and_reset_comm_stats():
    seq_rank = get_sequence_parallel_rank()

    global _fwd_send_volume, _fwd_recv_volume, _bwd_send_volume, _bwd_recv_volume
    _fwd_send_volume *= 1e-9
    _fwd_recv_volume *= 1e-9
    _bwd_send_volume *= 1e-9
    _bwd_recv_volume *= 1e-9

    print(f"R={seq_rank} fwd send: {_fwd_send_volume} fwd recv: {_fwd_recv_volume}; bwd send: {_bwd_send_volume}, bwd recv: {_bwd_recv_volume} GB.")
    _fwd_send_volume = 0
    _fwd_recv_volume = 0
    _bwd_send_volume = 0
    _bwd_recv_volume = 0

def launch_async_handles(handles, comm_mode):
    global _args
    if comm_mode == "nocomm":
        #print("skipping communication for ablation")
        return []
    if len(handles) > 0:
        return dist.batch_isend_irecv(handles)
    return []

def wait_async_handles(reqs):
    if len(reqs) > 0:
        for req in reqs:
            for r in req:
                r.wait()