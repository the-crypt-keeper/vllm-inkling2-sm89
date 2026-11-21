# SPDX-License-Identifier: Apache-2.0
"""Programmatic dependent launch (PDL) for the Inkling 2 kernels, off on sm89.

PDL is sm90+: `griddepcontrol.wait` / `launch_dependents` do not assemble for sm_89, and the
launch attribute needs Hopper. On sm89 the kernels run in plain stream order instead, where
the wait has nothing to wait for: `gdc_wait` / `gdc_launch_dependents` compile to nothing and
`launch_pdl` is False. Everywhere else this is the PR's behaviour unchanged.
"""

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

PDL: bool = not current_platform.is_device_capability(89)
_PDL = tl.constexpr(PDL)


@triton.jit
def gdc_wait():
    if _PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _PDL:
        tl.extra.cuda.gdc_launch_dependents()
