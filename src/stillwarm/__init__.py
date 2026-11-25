# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""stillwarm — automatic KV-cache session persistence for llama-server.

Your local LLM re-reads everything after every restart; stillwarm saves the
model's per-conversation "notes" (the KV cache) to disk and restores them —
with an integrity sidecar so an incompatible file is refused, never misloaded.
"""

from . import store
from .compat import ADVISORY, OK, REFUSE, Decision, ServerProfile, check, check_build
from .server import PROBE_N_PREDICT, PROBE_SAMPLER, LlamaServer, ServerError
from .session import (
    AskResult,
    IneffectiveRestoreWarning,
    RefusedRestoreError,
    Session,
    VerifyFailedError,
    chat_probe_body,
)
from .sidecar import SIDECAR_VERSION, Sidecar, sidecar_path_for

__version__ = "0.4.0"
__all__ = [
    "ADVISORY",
    "OK",
    "PROBE_N_PREDICT",
    "PROBE_SAMPLER",
    "REFUSE",
    "SIDECAR_VERSION",
    "AskResult",
    "Decision",
    "IneffectiveRestoreWarning",
    "LlamaServer",
    "RefusedRestoreError",
    "ServerError",
    "ServerProfile",
    "Session",
    "Sidecar",
    "VerifyFailedError",
    "__version__",
    "chat_probe_body",
    "check",
    "check_build",
    "sidecar_path_for",
    "store",
]
