import math

import triton
import triton.language as tl

import torch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + stride_qb * batch_index, 
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + stride_kb * batch_index,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + stride_vb * batch_index,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + stride_ob * batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + stride_lb * batch_index,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (bq, d)
    oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)   # (bq, d)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (bq,)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)   # (bq,)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (bk, d)
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (bk, d)
        
        sij = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offsets = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            sij = tl.where(causal_mask, sij, float("-inf"))
        mij = tl.maximum(mi, tl.max(sij, axis=1))
        pij = tl.exp(sij - mij[:, None])

        alpha = tl.exp(mi - mij)
        li = li * alpha + tl.sum(pij, axis=1)
        oi = alpha[:, None] * oi + tl.dot(pij, v)
        mi = mij

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    tl.store(O_block_ptr, oi / li[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, mi + tl.log(li), boundary_check=(0,))

class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q: (batch_size, Nq, d)
        K: (batch_size, Nk, d)
        V: (batch_size, Nk, d)
        """
        batch_size, Nq, d = Q.shape
        Nk = K.shape[1]
        O = torch.empty((batch_size, Nq, d), device=Q.device)
        L = torch.empty((batch_size, Nq,), device=Q.device)
        
        ctx.K_TILE_SIZE = 16
        ctx.Q_TILE_SIZE = 16
        # #region agent log
        import json as _json, os as _os; _os.makedirs("/home/lockzhou/workspace/cs336/assignment2-systems/.cursor", exist_ok=True); _lf = open("/home/lockzhou/workspace/cs336/assignment2-systems/.cursor/debug-e177ef.log", "a"); _lf.write(_json.dumps({"sessionId":"e177ef","hypothesisId":"H6","location":"triton_flash_attention.py:forward","message":"scale and params","data":{"scale":1.0/math.sqrt(d),"d":d,"batch_size":batch_size,"Nq":Nq,"Nk":Nk,"Q_dtype":str(Q.dtype)},"runId":"post-fix-scale","timestamp":__import__('time').time()}) + "\n"); _lf.close()
        # #endregion
        flash_fwd_kernel[(Nq // ctx.Q_TILE_SIZE, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(d), d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(Q, K, V, L, O)
        return O
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass not implemented yet")