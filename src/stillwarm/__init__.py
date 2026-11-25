"""stillwarm — automatic KV-cache session persistence for llama-server.

Your local LLM re-reads everything after every restart; stillwarm saves the
model's per-conversation "notes" (the KV cache) to disk and restores them —
with an integrity sidecar so an incompatible file is refused, never misloaded.
"""

from . import store
from .compat import ADVISORY, OK, REFUSE, Decision, ServerProfile, check, check_build
from .server import PROBE_N_PREDICT, PROBE_SAMPLER, LlamaServer, ServerError
from .session import AskResult, IneffectiveRestoreWarning, RefusedRestore, Session, VerifyFailed
from .sidecar import SIDECAR_VERSION, Sidecar, sidecar_path_for

__version__ = "0.1.1"
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
    "RefusedRestore",
    "ServerError",
    "ServerProfile",
    "Session",
    "Sidecar",
    "VerifyFailed",
    "__version__",
    "check",
    "check_build",
    "sidecar_path_for",
    "store",
]
