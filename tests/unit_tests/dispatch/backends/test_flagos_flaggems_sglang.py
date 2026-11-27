"""CPU contract tests for the FlagGems-sglang adapters in the FlagOS backend."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from sglang_fl import utils
from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
from sglang_fl.dispatch.backends.flagos.impl import activation, fla, flaggems_sglang
from sglang_fl.dispatch.backends.flagos.impl import gemma_rms_norm as gemma
from sglang_fl.dispatch.backends.flagos.register_ops import register_builtins
from sglang_fl.dispatch.backends.reference.impl.gemma_rms_norm import (
    gemma_rms_norm_torch,
)
from sglang_fl.dispatch.registry import OpRegistry


# Kernels mirroring flaggems_sglang/reference/*.py.
def _gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd * (1.0 + weight.float())).to(x.dtype)


def _gemma_fused_add_rmsnorm(x, residual, weight, eps):
    s = x.float() + residual.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype), s.to(x.dtype)


def _silu_and_mul(x):
    d = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()).to(
        x.dtype
    )


@pytest.fixture
def fake_package(monkeypatch):
    calls = []

    def recorded(name, fn):
        def wrapper(*args):
            calls.append((name, args))
            return fn(*args)

        return wrapper

    kernels = {
        "gemma_rmsnorm": recorded("gemma_rmsnorm", _gemma_rmsnorm),
        "gemma_fused_add_rmsnorm": recorded(
            "gemma_fused_add_rmsnorm", _gemma_fused_add_rmsnorm
        ),
        "silu_and_mul": recorded("silu_and_mul", _silu_and_mul),
    }
    package = ModuleType("flaggems_sglang")
    package.vendor_name = "nvidia"
    package.get_op = kernels.get
    package.calls = calls
    monkeypatch.setitem(sys.modules, "flaggems_sglang", package)
    monkeypatch.setattr(utils, "get_device_info", lambda: package)
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()
    yield package
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()


def _norm(hidden, dtype=torch.bfloat16):
    obj = torch.nn.Module()
    obj.weight = torch.nn.Parameter(torch.randn(hidden).to(dtype) * 0.5)
    obj.variance_epsilon = 1e-6
    return obj


@pytest.mark.parametrize("shape", [(5, 256), (2, 3, 256), (1, 2048)])
def test_gemma_without_residual_matches_reference(fake_package, shape):
    torch.manual_seed(0)
    obj = _norm(shape[-1])
    x = torch.randn(*shape).to(torch.bfloat16)

    actual = gemma.gemma_rms_norm_flagos(obj, x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, gemma_rms_norm_torch(obj, x))
    (name, (_, weight, eps)), = fake_package.calls
    assert name == "gemma_rmsnorm"
    assert weight is obj.weight
    assert eps == obj.variance_epsilon


def test_gemma_without_residual_accepts_strided_input(fake_package):
    torch.manual_seed(1)
    obj = _norm(256)
    qkv = torch.randn(4, 3 * 256).to(torch.bfloat16)
    k = qkv[:, 256:512]

    actual = gemma.gemma_rms_norm_flagos(obj, k)

    torch.testing.assert_close(actual, gemma_rms_norm_torch(obj, k))
    (_, (x2d, _, _)), = fake_package.calls
    assert x2d.is_contiguous()


@pytest.mark.parametrize("shape", [(7, 128), (2, 3, 512)])
def test_gemma_with_residual_matches_reference(fake_package, shape):
    torch.manual_seed(2)
    obj = _norm(shape[-1])
    x = torch.randn(*shape).to(torch.bfloat16)
    residual = torch.randn(*shape).to(torch.bfloat16)
    x_before, residual_before = x.clone(), residual.clone()

    out, residual_out = gemma.gemma_rms_norm_flagos(obj, x, residual)

    expected_out, expected_residual = gemma_rms_norm_torch(
        obj, x_before.clone(), residual_before.clone()
    )
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(residual_out, expected_residual)
    assert out.shape == residual_out.shape == x.shape
    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(residual, residual_before)
    (name, (_, _, weight, _)), = fake_package.calls
    assert name == "gemma_fused_add_rmsnorm"
    assert weight is obj.weight


def test_gemma_empty_input_skips_kernels(fake_package):
    obj = _norm(256)
    x = torch.empty(0, 256, dtype=torch.bfloat16)
    residual = torch.empty_like(x)

    assert gemma.gemma_rms_norm_flagos(obj, x) is x
    out, residual_out = gemma.gemma_rms_norm_flagos(obj, x, residual)
    assert out is x and residual_out is residual
    assert fake_package.calls == []


def test_gemma_follows_in_place_weight_updates(fake_package):
    obj = _norm(64)
    x = torch.randn(3, 64).to(torch.bfloat16)
    before = gemma.gemma_rms_norm_flagos(obj, x)

    with torch.no_grad():
        obj.weight.add_(1.0)

    after = gemma.gemma_rms_norm_flagos(obj, x)
    assert not torch.equal(before, after)
    torch.testing.assert_close(after, gemma_rms_norm_torch(obj, x))


def test_silu_and_mul_uses_flaggems_sglang(fake_package):
    x = torch.randn(3, 2, 16).to(torch.bfloat16)

    torch.testing.assert_close(activation.silu_and_mul_flagos(None, x), _silu_and_mul(x))
    assert [name for name, _ in fake_package.calls] == ["silu_and_mul"]


def _install_fake_flag_gems(monkeypatch, calls):
    module = ModuleType("flag_gems.modules.activation")

    def gems_silu_and_mul(x1, x2):
        calls.append("flag_gems")
        return torch.nn.functional.silu(x1) * x2

    module.gems_silu_and_mul = gems_silu_and_mul
    monkeypatch.setitem(sys.modules, "flag_gems", ModuleType("flag_gems"))
    monkeypatch.setitem(sys.modules, "flag_gems.modules", ModuleType("flag_gems.modules"))
    monkeypatch.setitem(sys.modules, "flag_gems.modules.activation", module)


def test_unlisted_vendor_falls_back(fake_package, monkeypatch):
    fake_package.vendor_name = "enflame"
    flag_gems_calls = []
    _install_fake_flag_gems(monkeypatch, flag_gems_calls)

    assert flaggems_sglang.get_op("silu_and_mul") is None
    assert not gemma.is_gemma_rms_norm_available()
    activation.silu_and_mul_flagos(None, torch.randn(2, 8))
    assert flag_gems_calls == ["flag_gems"]
    assert fake_package.calls == []


@pytest.mark.parametrize("missing_op", ["gemma_rmsnorm", "gemma_fused_add_rmsnorm"])
def test_missing_kernel_disables_gemma(fake_package, missing_op):
    lookup = fake_package.get_op
    fake_package.get_op = lambda name: None if name == missing_op else lookup(name)

    assert not gemma.is_gemma_rms_norm_available()


def test_broken_package_import_falls_back(monkeypatch):
    class _Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "flaggems_sglang":
                raise RuntimeError("op module failed to import")
            return None

    monkeypatch.delitem(sys.modules, "flaggems_sglang", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Broken(), *sys.meta_path])
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()
    try:
        assert flaggems_sglang.get_op("silu_and_mul") is None
    finally:
        flaggems_sglang._package.cache_clear()
        flaggems_sglang.get_op.cache_clear()


@pytest.mark.parametrize("vendor_name,expected", [("nvidia", True), ("enflame", False)])
def test_flagos_registers_gemma_rms_norm(fake_package, monkeypatch, vendor_name, expected):
    fake_package.vendor_name = vendor_name
    monkeypatch.setattr(FlagOSBackend, "_available", True)
    registry = OpRegistry()
    register_builtins(registry)

    (impl,) = registry.snapshot().impls_by_op["gemma_rms_norm"]

    assert impl.impl_id == "default.flagos"
    assert impl.is_available() is expected
    if expected:
        obj = _norm(32)
        x = torch.randn(2, 32).to(torch.bfloat16)
        torch.testing.assert_close(impl.fn(obj, x), gemma_rms_norm_torch(obj, x))


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_ragged_history_cache_and_grouped_heads(fake_package, monkeypatch, head_first):
    calls = []

    def kernel(q, k, v, initial_state, output_final_state, **kwargs):
        assert q.shape[2] == v.shape[2] == 2
        assert initial_state.shape == (1, 2, 4, 3)
        assert initial_state.is_contiguous() and initial_state.dtype == torch.float32
        assert output_final_state and kwargs["use_qk_l2norm_in_kernel"]
        calls.append(q.shape[1])
        return v, initial_state + q.shape[1]

    def get_op(name):
        assert name == "chunk_gated_delta_rule_fwd"
        return kernel

    monkeypatch.setattr(fake_package, "get_op", get_op)
    monkeypatch.setattr(FlagOSBackend, "is_available", lambda self: False)
    monkeypatch.setattr(utils, "get_device_info", lambda: None)
    registry = OpRegistry()
    register_builtins(registry)
    (impl,) = registry.snapshot().impls_by_op["chunk_gated_delta_rule"]
    assert impl.impl_id == "default.flagos"
    assert FlagOSBackend().is_op_available("chunk_gated_delta_rule")
    assert impl.is_available()

    length = 68
    q = torch.ones(1, length, 1, 4, dtype=torch.bfloat16)
    v = torch.randn(1, length, 2, 3, dtype=torch.bfloat16)
    gates = torch.ones(1, length, 2)
    args = (q, q, v, gates, gates)
    if head_first:
        args = tuple(x.transpose(1, 2) for x in args)
    pool = torch.arange(4 * 2 * 3 * 4).reshape(4, 2, 3, 4).float()
    before = pool.clone()
    out, final, h = impl.fn(
        *args,
        0.5,
        pool,
        torch.tensor([0, -1, 1, 2]),
        torch.tensor([0, 65, 66, 66, 68]),
        head_first,
        True,
    )
    assert calls == [64, 1, 1, 2]
    assert final is None
    torch.testing.assert_close(out, args[2])
    expected_h = torch.stack(
        (before[0], before[0] + 64, torch.zeros_like(before[0]), before[2])
    )
    torch.testing.assert_close(h, expected_h.unsqueeze(0).to(q.dtype))
    before[0] += 65
    before[2] += 2
    torch.testing.assert_close(pool, before)


@pytest.mark.parametrize("sglang_kernel", [None, object()])
def test_listed_vendor_requires_callable_chunk_export(fake_package, monkeypatch, sglang_kernel):
    monkeypatch.setattr(fake_package, "get_op", lambda name: sglang_kernel)
    with pytest.raises(ImportError, match="chunk_gated_delta_rule_fwd"):
        flaggems_sglang.get_op("chunk_gated_delta_rule_fwd")
    monkeypatch.setattr(fla, "_chunk_kernels", lambda: pytest.fail("unexpected legacy lookup"))
    registry = OpRegistry()
    register_builtins(registry)
    (impl,) = registry.snapshot().impls_by_op["chunk_gated_delta_rule"]
    assert not impl.is_available()
    x = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ImportError, match="chunk_gated_delta_rule_fwd"):
        impl.fn(x, x, x, x[..., 0], x[..., 0], 1.0)


@pytest.mark.parametrize("vendor_name", ["kunlunxin", "hygon"])
def test_unlisted_vendor_checks_legacy_chunk_kernels(fake_package, monkeypatch, vendor_name):
    fake_package.vendor_name = vendor_name
    monkeypatch.setattr(fake_package, "get_op", lambda name: pytest.fail("unexpected fused lookup"))
    assert flaggems_sglang.get_op("chunk_gated_delta_rule_fwd") is None
    monkeypatch.delattr(torch, "corex", raising=False)
    monkeypatch.setattr(fla, "_chunk_kernels", lambda: (lambda: None,) * 4)
    registry = OpRegistry()
    register_builtins(registry)
    (impl,) = registry.snapshot().impls_by_op["chunk_gated_delta_rule"]
    backend = FlagOSBackend()
    assert backend.is_op_available("chunk_gated_delta_rule")
    assert impl.is_available()
    assert fla.is_fla_op_available("chunk_gated_delta_rule")

    monkeypatch.setattr(fla, "_chunk_kernels", lambda: (None,) * 4)
    assert not backend.is_op_available("chunk_gated_delta_rule")
    assert not fla.is_fla_op_available("chunk_gated_delta_rule")
    assert not impl.is_available()


@pytest.mark.parametrize("vendor_name,expected", [("iluvatar", False), ("kunlunxin", True)])
def test_chunk_without_sglang_package(fake_package, monkeypatch, vendor_name, expected):
    fake_package.vendor_name = vendor_name
    monkeypatch.setitem(sys.modules, "flaggems_sglang", None)
    monkeypatch.setattr(fla, "_chunk_kernels", lambda: (lambda: None,) * 4)
    assert FlagOSBackend().is_op_available("chunk_gated_delta_rule") is expected


@pytest.mark.parametrize("vendor_name", ["nvidia", "iluvatar", "thead", "mthreads", "metax"])
def test_chunk_dense_keeps_history_per_request(fake_package, monkeypatch, vendor_name):
    fake_package.vendor_name = vendor_name

    def kernel(*, v, initial_state, **kwargs):
        return v, initial_state + v.sum(dim=1).unsqueeze(-2)

    monkeypatch.setattr(fake_package, "get_op", lambda name: kernel)
    q = torch.ones(2, 65, 1, 4, dtype=torch.bfloat16)
    v = torch.ones(2, 65, 1, 3, dtype=q.dtype)
    v[1].fill_(2)
    gate = torch.ones(2, 65, 1)
    out, final, h = fla.chunk_gated_delta_rule_flagos(q, q, v, gate, gate, 0.5)
    torch.testing.assert_close(out, v)
    assert final is None and h.shape == (2, 2, 1, 3, 4)
    assert h[:, 0].eq(0).all()
    assert h[0, 1].eq(64).all() and h[1, 1].eq(128).all()
