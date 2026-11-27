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

# Loader for FlagGems-sglang (``flaggems_sglang``) fused kernels.

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Vendors on which the FlagGems-sglang maintainers declared each kernel ready.
# flaggems_sglang resolves a generic or vendor kernel for every vendor it can
# detect, so importability alone does not mean the kernel was validated there.
_COMMON_READY_VENDORS = frozenset(
    {"nvidia", "iluvatar", "hygon", "kunlunxin", "mthreads", "ascend", "thead", "metax"}
)
_READY_VENDORS = {
    "gemma_rmsnorm": _COMMON_READY_VENDORS,
    "gemma_fused_add_rmsnorm": _COMMON_READY_VENDORS,
    "silu_and_mul": _COMMON_READY_VENDORS,
    "chunk_gated_delta_rule_fwd": frozenset({"nvidia", "iluvatar", "thead", "mthreads", "metax"}),
}


@functools.lru_cache(maxsize=1)
def _package():
    try:
        import flaggems_sglang
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001
        # Importing the package imports every op module; one broken module
        # makes the whole package unusable on this vendor.
        logger.warning("flaggems_sglang import failed, falling back: %s", e)
        return None
    return flaggems_sglang


@functools.lru_cache(maxsize=None)
def get_op(name: str) -> Optional[Callable]:
    """Resolve a kernel. Listed chunk vendors require a callable export."""
    vendors = _READY_VENDORS.get(name, ())
    require_kernel = name == "chunk_gated_delta_rule_fwd"
    if require_kernel:
        from sglang_fl.utils import get_device_info

        device = get_device_info() or _package()
        if device is None:
            raise ImportError(f"Cannot determine the device vendor for {name}")
        if device.vendor_name not in vendors:
            return None

    package = _package()
    kernel = None
    if package is not None and package.vendor_name in vendors:
        kernel = package.get_op(name)
    if require_kernel and not callable(kernel):
        raise ImportError(f"FlagGems-sglang kernel {name} is unavailable for this device")
    return kernel
