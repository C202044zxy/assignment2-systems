import torch

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q: (batch_size, Nq, d)
        K: (batch_size, Nk, d)
        V: (batch_size, Nk, d)
        """
        print(Q.shape)
        Nq = Q.shape[0]
        d = Q.shape[1]
        Nk = K.shape[0]
        device = Q.device
        dtype = Q.dtype

        bq = 16
        bk = 16
        O = torch.zeros_like(Q)
        L = torch.zeros(Nq, device=device, dtype=dtype)
        for i in range(0, Nq, bq):
            q = Q[i: i + bq]   # (bq, d)
            oi = torch.zeros_like(q)   # (bq, d)
            li = torch.zeros(q.shape[0], device=device, dtype=dtype)
            mi = torch.full((q.shape[0],), -float("inf"), device=device, dtype=dtype)

            for j in range(0, Nk, bk):
                k = K[j: j + bk]
                v = V[j: j + bk]

                print(q.shape)
                print(k.shape)
                sij = (q @ k.T) / torch.sqrt(d) # (bq, bk)
                mij = torch.maximum(mi, torch.max(sij, dim = 1).values) # (bq, )
                pij = torch.exp(sij - mij[:, None]) # (bq, bk)

                alpha = torch.exp(mi - mij)
                li = alpha * li + torch.sum(pij, dim = 1)
                oi = alpha[:, None] * oi + (pij @ v)

            O[i: i + bq] = oi / li[:, None]
            L[i: i + bq] = mi + torch.log(li)
        ctx.save_for_backward(Q, K, V, L, O)
        return O
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass not implemented yet")