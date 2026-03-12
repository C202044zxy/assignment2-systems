import math

import torch


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q: (batch_size, Nq, d)
        K: (batch_size, Nk, d)
        V: (batch_size, Nk, d)
        """
        batch_size, Nq, d = Q.shape
        Nk = K.shape[1]
        device = Q.device
        dtype = Q.dtype
        scale = math.sqrt(d)

        bq = 16
        bk = 16
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, Nq, device=device, dtype=dtype)

        for b in range(batch_size):
            for i in range(0, Nq, bq):
                q = Q[b, i:i + bq]       # (bq, d)
                cur_bq = q.shape[0]
                oi = torch.zeros_like(q)  # (bq, d)
                li = torch.zeros(cur_bq, device=device, dtype=dtype)
                mi = torch.full((cur_bq,), -float("inf"), device=device, dtype=dtype)

                for j in range(0, Nk, bk):
                    k = K[b, j:j + bk]   # (bk, d)
                    v = V[b, j:j + bk]   # (bk, d)

                    sij = (q @ k.T) / scale                          # (bq, bk)

                    mij = torch.maximum(mi, sij.max(dim=1).values)   # (bq,)
                    pij = torch.exp(sij - mij[:, None])              # (bq, bk)

                    alpha = torch.exp(mi - mij)
                    li = alpha * li + pij.sum(dim=1)
                    oi = alpha[:, None] * oi + (pij @ v)

                    mi = mij

                O[b, i:i + bq] = oi / li[:, None]
                L[b, i:i + bq] = mi + torch.log(li)

        ctx.save_for_backward(Q, K, V, L, O)
        return O
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass not implemented yet")