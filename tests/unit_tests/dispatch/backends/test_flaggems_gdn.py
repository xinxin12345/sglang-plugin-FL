# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagGems-sglang provider selection and state ownership, without a device."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
from sglang_fl.dispatch.backends.flagos.impl import fla, flaggems_gdn


@pytest.mark.parametrize(
    "vendor,available",
    [
        ("nvidia", (True, True)),
        ("iluvatar", (True, True)),
        ("thead", (True, True)),
        ("metax", (True, False)),
        ("mthreads", (True, False)),
        ("kunlunxin", (False, False)),
    ],
)
def test_provider_selection(monkeypatch, vendor, available):
    names = []

    def lookup(name):
        names.append(name)
        return lambda: None

    package = SimpleNamespace(vendor_name=vendor, get_op=lookup)
    monkeypatch.setitem(sys.modules, "flaggems_sglang", package)
    monkeypatch.setattr(FlagOSBackend, "is_available", lambda self: False)
    get_op = flaggems_gdn.get_gdn_op
    get_op.cache_clear()
    try:
        ops = (flaggems_gdn.CHUNK, flaggems_gdn.PACKED)
        assert tuple(FlagOSBackend().is_op_available(op) for op in ops) == available
        assert names == [
            op
            for op, enabled in zip(("chunk_gated_delta_rule_fwd", ops[1]), available)
            if enabled
        ]
        package.get_op = lambda name: None
        get_op.cache_clear()
        assert all(not FlagOSBackend().is_op_available(op) for op in ops)
    finally:
        get_op.cache_clear()


def test_packed_remaps_zero_without_mutating_other_slots(monkeypatch):
    from sglang_fl.dispatch.backends.flagos.impl import gdn_state

    def kernel(*, initial_state, ssm_state_indices, out, **kwargs):
        assert ssm_state_indices.tolist() == [1, -1, 3]
        assert kwargs["a"].is_contiguous() and kwargs["b"].is_contiguous()
        assert initial_state[1].eq(3).all() and initial_state[3].eq(7).all()
        initial_state[1].fill_(13)
        initial_state[3].fill_(17)
        out.fill_(2)
        out[1].zero_()
        return out, initial_state

    def scatter(pool, states, slots):
        valid = slots >= 0
        pool.index_copy_(0, slots[valid].long(), states[valid])

    monkeypatch.setattr(flaggems_gdn, "get_gdn_op", lambda name: kernel)
    monkeypatch.setattr(gdn_state, "store_decode_states", scatter)
    pool = torch.full((4, 2, 3, 4), 5.0)
    pool[0].fill_(3)
    pool[2].fill_(7)
    out = torch.empty(3, 1, 2, 3)
    a = torch.ones(3, 4)[:, :2]
    result, state = fla.fused_recurrent_gated_delta_rule_packed_decode_flagos(
        torch.ones(3, 14),
        a,
        a,
        torch.ones(2),
        torch.ones(2),
        0.5,
        pool,
        out,
        torch.tensor([0, -1, 2]),
        True,
    )
    assert result is out and state is pool
    assert out[:, 0, 0, 0].tolist() == [2, 0, 2]
    assert pool[:, 0, 0, 0].tolist() == [13, 5, 17, 5]


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_ragged_history_cache_and_grouped_heads(monkeypatch, head_first):
    calls = []

    def kernel(q, k, v, initial_state, output_final_state, **kwargs):
        assert q.shape[2] == v.shape[2] == 2
        assert initial_state.shape == (1, 2, 4, 3)
        assert initial_state.is_contiguous() and initial_state.dtype == torch.float32
        assert output_final_state and kwargs["use_qk_l2norm_in_kernel"]
        calls.append(q.shape[1])
        return v, initial_state + q.shape[1]

    monkeypatch.setattr(flaggems_gdn, "get_gdn_op", lambda name: kernel)
    length = 68
    q = torch.ones(1, length, 1, 4)
    v = torch.randn(1, length, 2, 3)
    gates = torch.ones(1, length, 2)
    args = (q, q, v, gates, gates)
    if head_first:
        args = tuple(x.transpose(1, 2) for x in args)
    pool = torch.arange(4 * 2 * 3 * 4).reshape(4, 2, 3, 4).float()
    before = pool.clone()
    out, final, h = fla.chunk_gated_delta_rule_flagos(
        *args,
        0.5,
        pool,
        torch.tensor([0, -1, 2]),
        torch.tensor([0, 65, 66, 68]),
        head_first,
        True,
    )
    assert calls == [64, 1, 1, 2]
    assert final is None
    torch.testing.assert_close(out, args[2])
    expected_h = torch.stack(
        (before[0], before[0] + 64, torch.zeros_like(before[0]), before[2])
    )
    torch.testing.assert_close(h, expected_h.unsqueeze(0))
    before[0] += 65
    before[2] += 2
    torch.testing.assert_close(pool, before)
