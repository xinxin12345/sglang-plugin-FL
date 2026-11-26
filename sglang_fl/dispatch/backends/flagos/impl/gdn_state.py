# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe writeback from a compact decode batch to a strided state pool."""

import triton
import triton.language as tl


@triton.jit
def _store_states(
    src, dst, slots, dst_stride, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(src + row.to(tl.int64) * WIDTH + offsets, mask=offsets < WIDTH)
        tl.store(dst + slot * dst_stride + offsets, values, mask=offsets < WIDTH)


def store_decode_states(pool, states, slots):
    width = states.stride(0)
    _store_states[(states.shape[0], triton.cdiv(width, 256))](
        states,
        pool,
        slots,
        pool.stride(0),
        WIDTH=width,
        BLOCK=256,
    )
