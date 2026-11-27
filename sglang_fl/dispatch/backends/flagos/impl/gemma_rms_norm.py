# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FlagOS GemmaRMSNorm implementation using FlagGems-sglang kernels.
# GemmaRMSNorm: output = rms_norm(x) * (weight + 1.0)

from __future__ import annotations

from typing import Optional, Union

import torch

from .flaggems_sglang import get_op


def is_gemma_rms_norm_available() -> bool:
    return (
        get_op("gemma_rmsnorm") is not None
        and get_op("gemma_fused_add_rmsnorm") is not None
    )


def gemma_rms_norm_flagos(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Gemma-style RMS normalization using FlagGems-sglang.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, x + residual) if residual
        provided
    """
    if x.numel() == 0:
        return x if residual is None else (x, residual)

    # Vendor kernels address rows as a dense [M, D] matrix.
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    eps = obj.variance_epsilon

    if residual is None:
        out = get_op("gemma_rmsnorm")(x2d, obj.weight, eps)
        return out.view(shape)

    out, residual_out = get_op("gemma_fused_add_rmsnorm")(
        x2d,
        residual.reshape(-1, shape[-1]).contiguous(),
        obj.weight,
        eps,
    )
    return out.view(shape), residual_out.view(residual.shape)
