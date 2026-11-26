# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Adapt FlagGems-sglang's GDN operators to the SGLang 0.5.18 state cache."""

import logging
from functools import cache
from itertools import pairwise

import torch

logger = logging.getLogger(__name__)

PACKED = "fused_recurrent_gated_delta_rule_packed_decode"
CHUNK = "chunk_gated_delta_rule"
_PROVIDERS = {
    PACKED: (PACKED, {"nvidia", "iluvatar", "thead"}),
    CHUNK: (
        "chunk_gated_delta_rule_fwd",
        {"nvidia", "iluvatar", "thead", "mthreads", "metax"},
    ),
}


@cache
def get_gdn_op(name):
    entry = _PROVIDERS.get(name)
    if entry is None:
        return None
    try:
        import flaggems_sglang

        provider_name, vendors = entry
        if flaggems_sglang.vendor_name in vendors:
            fn = flaggems_sglang.get_op(provider_name)
            return fn if callable(fn) else None
    except ImportError:
        pass
    except Exception:
        # Some optional backends fail while loading their device libraries.
        logger.warning("Could not resolve FlagGems-sglang %s", name, exc_info=True)
    return None


def packed_decode(
    kernel, mixed_qkv, a, b, A_log, dt_bias, scale, pool, out, slots, normalize
):
    from .gdn_state import store_decode_states

    batch = mixed_qkv.shape[0]
    # Slot zero is live in 0.5.18, but is padding in FlagGems-sglang.
    # Give each live request a private positive slot. Only stage this batch.
    work = pool.new_empty((batch + 1, *pool.shape[1:]))
    slots = slots.contiguous()
    torch.index_select(pool, 0, slots.clamp_min(0).long(), out=work[1:])
    local_slots = torch.arange(1, batch + 1, device=slots.device, dtype=slots.dtype)
    local_slots = torch.where(slots >= 0, local_slots, -1)
    kernel(
        mixed_qkv=mixed_qkv.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=A_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        scale=scale,
        initial_state=work,
        out=out,
        ssm_state_indices=local_slots,
        use_qk_l2norm_in_kernel=normalize,
    )
    store_decode_states(pool, work[1:], slots)
    return out, pool


def _run_chunks(kernel, inputs, state, scale, normalize):
    """Process a dense batch, preserving the state at the start of each block."""
    q, k, v, g, beta = inputs
    batch, length, hv, value_dim = v.shape
    state = state.transpose(-1, -2).float().contiguous()
    blocks = (length + 63) // 64
    history = k.new_empty((batch, blocks, hv, value_dim, k.shape[-1]))
    output = torch.empty_like(v)
    for index in range(blocks):
        span = slice(64 * index, 64 * (index + 1))
        history[:, index].copy_(state.transpose(-1, -2))
        block_out, state = kernel(
            q=q[:, span].contiguous(),
            k=k[:, span].contiguous(),
            v=v[:, span].contiguous(),
            g=g[:, span].contiguous(),
            beta=beta[:, span].contiguous(),
            scale=scale,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=normalize,
        )
        output[:, span].copy_(block_out)
    return output, state.transpose(-1, -2), history


def chunk(kernel, inputs, state, scale, cu_seqlens, normalize):
    q, k, v, g, beta = inputs
    query_heads, value_heads = q.shape[2], v.shape[2]
    if query_heads <= 0 or value_heads % query_heads:
        raise ValueError("value heads must be divisible by query heads")
    if query_heads != value_heads:
        groups = value_heads // query_heads
        q = q.repeat_interleave(groups, dim=2)
        k = k.repeat_interleave(groups, dim=2)
    inputs = (q, k, v, g, beta)
    if state is None:
        count = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        state = v.new_zeros(
            (count, value_heads, v.shape[-1], k.shape[-1]), dtype=torch.float32
        )
    if cu_seqlens is None or cu_seqlens.numel() == 2:
        return _run_chunks(kernel, inputs, state, scale, normalize)

    # The packed history concatenates each sequence's ceil(length / 64) entries.
    bounds = cu_seqlens.tolist()
    if not bounds or bounds[0] != 0 or bounds[-1] != q.shape[1]:
        raise ValueError("cu_seqlens must span all packed tokens")
    outputs, finals, histories = [], [], []
    for i, (start, end) in enumerate(pairwise(bounds)):
        if end < start:
            raise ValueError("cu_seqlens must be nondecreasing")
        seq = tuple(x[:, start:end] for x in inputs)
        o, final, h = _run_chunks(kernel, seq, state[i : i + 1], scale, normalize)
        outputs.append(o)
        finals.append(final)
        histories.append(h)
    if not outputs:
        return (
            torch.empty_like(v),
            state,
            k.new_empty((1, 0, value_heads, v.shape[-1], k.shape[-1])),
        )
    return torch.cat(outputs, dim=1), torch.cat(finals), torch.cat(histories, dim=1)
