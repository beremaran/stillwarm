"""stillwarm — automatic KV-cache session persistence for llama-server.

Your local LLM re-reads everything after every restart; stillwarm saves the
model's per-conversation "notes" (the KV cache) to disk and restores them —
with an integrity sidecar so an incompatible file is refused, never misloaded.
"""
from .sidecar import Sidecar, sidecar_path_for, SIDECAR_VERSION
from .compat import ServerProfile, Decision, check, check_build, REFUSE, ADVISORY, OK
from .server import LlamaServer, ServerError, PROBE_SAMPLER, PROBE_N_PREDICT
from .session import Session, AskResult, RefusedRestore, VerifyFailed, IneffectiveRestoreWarning
from . import store

__version__ = "0.1.0"
__all__ = ["Sidecar", "sidecar_path_for", "SIDECAR_VERSION", "ServerProfile",
           "Decision", "check", "check_build", "REFUSE", "ADVISORY", "OK",
           "LlamaServer", "ServerError", "PROBE_SAMPLER", "PROBE_N_PREDICT",
           "Session", "AskResult", "RefusedRestore", "VerifyFailed",
           "IneffectiveRestoreWarning", "store", "__version__"]
