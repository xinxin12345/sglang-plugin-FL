# Copyright (c) 2026 BAAI. All rights reserved.
"""SGLang FLA contracts implemented with FlagGems 5.3.4 kernels."""

from __future__ import annotations

import inspect
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .flaggems_sglang import get_op


def _chunk_kernels():
    from flag_gems.fused.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
    from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
        chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
    )
    from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd

    return (
        chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
        recompute_w_u_fwd,
        chunk_gated_delta_rule_fwd_h,
        chunk_fwd_o,
    )


def _recurrent_kernel():
    from flag_gems.fused.FLA import fused_recurrent_gated_delta_rule_fwd

    return fused_recurrent_gated_delta_rule_fwd


def _packed_decode_kernel():
    import triton
    from flag_gems.fused.FLA.fused_recurrent import (
        fused_recurrent_gated_delta_rule_large_t_fwd_kernel,
    )

    return triton, fused_recurrent_gated_delta_rule_large_t_fwd_kernel


def is_fla_op_available(op_name: str) -> bool:
    """Check for a supported implementation of the requested FLA operator."""
    try:
        if op_name == "chunk_gated_delta_rule":
            return get_op("chunk_gated_delta_rule_fwd") is not None or all(
                callable(kernel) for kernel in _chunk_kernels()
            )
        if op_name == "fused_recurrent_gated_delta_rule_packed_decode":
            triton, kernel = _packed_decode_kernel()
            return hasattr(kernel, "__getitem__") and callable(triton.next_power_of_2)
        if op_name == "fused_recurrent_gated_delta_rule":
            kernel = _recurrent_kernel()
            return (
                callable(kernel)
                and "inplace_final_state" in inspect.signature(kernel).parameters
            )
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return False


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    # F.normalize clamps the norm; SGLang's FLA l2norm adds epsilon instead.
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def chunk_gated_delta_rule_flagos(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Return (output, None, chunk_states), updating indexed cache slots.

    SGLang stores states as [V, K], whereas FlagGems' chunk kernels use [K, V].
    The intermediate states must use SGLang's fixed 64-token chunk size too.
    """
    kernel = get_op("chunk_gated_delta_rule_fwd")
    if kernel is not None and (cu_seqlens is not None or initial_state_indices is not None):
        device = getattr(torch, q.device.type, None)
        capturing = getattr(device, "is_current_stream_capturing", None)
        if callable(capturing) and capturing():
            # Ragged history sizes and masked slot writes depend on device data.
            from sglang_fl.dispatch.fla_patch import get_original

            native = get_original("chunk_gated_delta_rule")
            if native is None:
                raise RuntimeError("Original SGLang chunk function unavailable during capture")
            return native(
                q, k, v, g, beta, scale, initial_state, initial_state_indices,
                cu_seqlens, head_first, use_qk_l2norm_in_kernel,
            )
    if head_first:
        q, k, v, g, beta = (x.transpose(1, 2) for x in (q, k, v, g, beta))
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1

    state = initial_state
    expected_state_shape = (v.shape[2], v.shape[-1], k.shape[-1])
    if state is not None and (
        state.ndim != 4 or state.shape[1:] != expected_state_shape
    ):
        raise ValueError(
            f"initial_state must have shape [slots, {expected_state_shape}]"
        )
    indices = valid = None
    if initial_state_indices is not None:
        if initial_state is None:
            raise ValueError("initial_state_indices requires initial_state")
        if initial_state_indices.shape != (num_sequences,):
            raise ValueError("initial_state_indices must have one index per sequence")
        indices = initial_state_indices.long()
        valid = indices >= 0
        state = initial_state.index_select(0, indices.clamp_min(0))
        state = state.masked_fill(~valid[:, None, None, None], 0)
    elif initial_state is not None and initial_state.shape[0] != num_sequences:
        raise ValueError("initial_state must have one state per sequence")
    if kernel is not None:
        batch, tokens, value_heads, value_dim = v.shape
        query_heads, key_dim = q.shape[2:]
        if k.shape[2] != query_heads or query_heads <= 0 or value_heads % query_heads:
            raise ValueError("Q/K heads must match and divide the number of value heads")
        if value_heads != query_heads:
            q = q.repeat_interleave(value_heads // query_heads, dim=2)
            k = k.repeat_interleave(value_heads // query_heads, dim=2)

        bounds = [0, tokens] if cu_seqlens is None else cu_seqlens.tolist()
        if not bounds or bounds[0] != 0 or bounds[-1] != tokens:
            raise ValueError("cu_seqlens must span all packed tokens")
        lengths = [end - start for start, end in zip(bounds, bounds[1:])]
        if any(length < 0 for length in lengths):
            raise ValueError("cu_seqlens must be nondecreasing")
        block_size = 64
        num_blocks = sum((length + block_size - 1) // block_size for length in lengths)
        history = k.new_empty((batch, num_blocks, value_heads, value_dim, key_dim))
        output = torch.empty_like(v)
        if state is None:
            state = v.new_zeros((num_sequences, value_heads, key_dim, value_dim), dtype=torch.float32)
        else:
            state = state.transpose(-1, -2).float().contiguous()
        final_state = torch.empty_like(state) if indices is not None else None

        # Dense requests share the token axis. Packed requests have independent boundaries.
        block_index = 0
        for seq, (start, end) in enumerate(zip(bounds, bounds[1:])):
            rows = slice(None) if cu_seqlens is None else slice(seq, seq + 1)
            block_state = state[rows]
            for pos in range(start, end, block_size):
                span = slice(pos, min(pos + block_size, end))
                history[:, block_index].copy_(block_state.transpose(-1, -2))
                block_output, block_state = kernel(
                    q=q[:, span].contiguous(),
                    k=k[:, span].contiguous(),
                    v=v[:, span].contiguous(),
                    g=g[:, span].contiguous(),
                    beta=beta[:, span].contiguous(),
                    scale=scale,
                    initial_state=block_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                output[:, span].copy_(block_output)
                block_index += 1
            if final_state is not None:
                final_state[rows].copy_(block_state)
        if indices is not None:
            updated = final_state[valid].transpose(-1, -2).to(initial_state.dtype).contiguous()
            initial_state.index_copy_(0, indices[valid], updated)
        if head_first:
            output = output.transpose(1, 2)
        return output, None, history

    cumsum_solve, recompute, recurrent, output_kernel = _chunk_kernels()
    if state is not None:
        state = state.transpose(-1, -2).contiguous()

    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
    scale = k.shape[-1] ** -0.5 if scale is None else float(scale)

    # The public FlagGems chunk wrapper suppresses h by default and can select
    # chunks smaller than 64. Compose its same subkernels explicitly so SGLang
    # can use h for radix state tracking without changing process-global flags.
    g_cumsum, A = cumsum_solve(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
        output_dtype=k.dtype,
    )
    w, u = recompute(k=k, v=v, beta=beta, A=A, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens)
    h, v_new, final_state = recurrent(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=state,
        output_final_state=True,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
    )
    output = output_kernel(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )
    if indices is not None:
        state_to_copy = final_state[valid].transpose(-1, -2).to(initial_state.dtype)
        if hasattr(torch, "corex"):
            # FlagGems index_copy_ derives destination offsets from source strides.
            state_to_copy = state_to_copy.contiguous()
        initial_state.index_copy_(0, indices[valid], state_to_copy)
    if head_first:
        output = output.transpose(1, 2)
    return output.to(q.dtype), None, h.transpose(-1, -2).contiguous()


def fused_recurrent_gated_delta_rule_flagos(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapt the public SGLang recurrent API without mutating its input state."""
    if ssm_state_indices is not None or num_accepted_tokens is not None:
        raise NotImplementedError(
            "FlagOS public recurrent op does not support legacy scheduling hints"
        )
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    batch, length = q.shape[:2]
    if cu_seqlens is None:
        num_sequences = batch
        cu_seqlens = torch.arange(batch + 1, device=q.device, dtype=torch.long) * length
    else:
        if batch != 1:
            raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
        num_sequences = cu_seqlens.numel() - 1
        cu_seqlens = cu_seqlens.contiguous()
    state_shape = (num_sequences, v.shape[2], v.shape[-1], k.shape[-1])
    if initial_state is None:
        state = q.new_zeros(state_shape, dtype=torch.float32)
    else:
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}")
        state = initial_state.clone()
    if length == 0:
        return torch.empty_like(v), state if output_final_state else None

    # In FlagGems 5.3.4 the short, strided Qwen kernel stores [K,V];
    # the packed-stride kernel also stores [K,V], and the long one [V,K]. Mirror its
    # dispatch condition exactly, including the H=1,T=2 contiguous case.
    packed_stride = all(x.stride(0) == x.stride(1) + x.stride(2) for x in (q, k, v))
    transpose_state = packed_stride or length <= 64
    if transpose_state:
        state = state.transpose(-1, -2).contiguous()
    # The low-level kernel writes once per token. Repeating each dense slot
    # keeps the final token's state while avoiding its per-token state output.
    indices = torch.arange(num_sequences, device=q.device, dtype=torch.long)
    indices = indices[:, None].repeat(1, length)
    output, final_state = _recurrent_kernel()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=k.shape[-1] ** -0.5 if scale is None else float(scale),
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if transpose_state:
        final_state = final_state.transpose(-1, -2).contiguous()
    return output, final_state if output_final_state else None


def fused_recurrent_gated_delta_rule_packed_decode_flagos(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run independent single-token requests and update caller-owned buffers."""
    if mixed_qkv.ndim != 2 or initial_state.ndim != 4:
        raise ValueError("packed decode requires 2D mixed_qkv and 4D initial_state")
    batch = mixed_qkv.shape[0]
    value_heads, value_dim, key_dim = initial_state.shape[-3:]
    if min(value_heads, value_dim, key_dim) <= 0:
        raise ValueError("packed decode head counts and dimensions must be positive")
    if initial_state.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("initial_state must be contiguous within each cache slot")
    if any(
        tensor.device != mixed_qkv.device
        for tensor in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    ):
        raise ValueError("all packed decode inputs must be on the same device")
    if not out.is_contiguous():
        raise ValueError("packed decode out must be contiguous")
    qk_dim = mixed_qkv.shape[-1] - value_heads * value_dim
    if qk_dim <= 0 or qk_dim % 2 or (qk_dim // 2) % key_dim:
        raise ValueError("invalid packed QKV dimension")
    query_dim = qk_dim // 2
    query_heads = query_dim // key_dim
    if value_heads % query_heads:
        raise ValueError("value heads must be divisible by query heads")
    if a.shape != (batch, value_heads) or b.shape != a.shape:
        raise ValueError("a and b must have shape [batch, value_heads]")
    if A_log.shape != (value_heads,) or dt_bias.shape != A_log.shape:
        raise ValueError("A_log and dt_bias must have shape [value_heads]")
    if ssm_state_indices.shape != (batch,) or out.shape != (
        batch,
        1,
        value_heads,
        value_dim,
    ):
        raise ValueError("invalid packed decode state indices or output shape")
    if batch == 0:
        return out, initial_state

    query, key, value = torch.split(
        mixed_qkv, (query_dim, query_dim, value_heads * value_dim), -1
    )
    # Use the [V,K] kernel directly: the public FlagGems wrapper selects a
    # short-sequence [K,V] kernel at T=1, which also lacks PAD_SLOT_ID guards.
    # Its large-T kernel implements the same recurrence for a single token
    # while safely reading/updating SGLang's live cache (including graph replay).
    query = query.reshape(batch, 1, query_heads, key_dim).contiguous()
    key = key.reshape(batch, 1, query_heads, key_dim).contiguous()
    value = value.reshape(batch, 1, value_heads, value_dim).contiguous()
    g = -A_log.float().exp() * F.softplus(
        a.float() + dt_bias.float(), beta=1.0, threshold=20.0
    )
    beta = b.float().sigmoid()
    triton, kernel = _packed_decode_kernel()
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (1, triton.cdiv(value_dim, block_v), batch * value_heads)
    kernel[grid](
        q=query,
        k=key,
        v=value,
        g=g[:, None].contiguous(),
        beta=beta[:, None].contiguous(),
        o=out,
        h0=initial_state,
        ht=initial_state,
        num_accepted_tokens=None,
        scale=scale,
        cu_seqlens=torch.arange(batch + 1, device=mixed_qkv.device, dtype=torch.long),
        ssm_state_indices=ssm_state_indices,
        N=batch,
        T=1,
        B=batch,
        H=query_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_indices_tok=1,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=True,
        IS_KDA=False,
        num_warps=1,
        num_stages=3,
    )
    # SGLang 0.5.18 writes zero output for PAD_SLOT_ID. FlagGems skips those
    # rows entirely, so finish that public contract without touching the cache.
    out.masked_fill_((ssm_state_indices < 0)[:, None, None, None], 0)
    return out, initial_state
