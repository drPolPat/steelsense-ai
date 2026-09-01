"""Workaround for a machine-specific Windows security policy, not a code bug.

On (at least) the machine this was developed on, a Windows application-
control policy intermittently blocks a handful of small, less-common
compiled Python extensions -- observed with `uuid_utils`, `xxhash`, and
`orjson` -- while much larger native packages this project also depends
on (numpy, pandas, onnxruntime, tokenizers) load without issue.

`uuid_utils` and `xxhash` are pulled in transitively by LangChain/LangGraph
only for optional features this project never uses (LangSmith tracing and
the hosted LangGraph Platform client) -- we run LangGraph's `StateGraph`
fully in-process, with no tracing configured. `orjson` is a direct,
required dependency of chromadb itself (used by the RAG layer), which is
why this module lives at the `src.backend` level rather than under
`agent/` -- both `rag/retrieval.py` and `agent/agent_langgraph.py` need it.

Each shim below is installed only if the real package actually fails to
import, so on a machine without this restriction the real, native-
optimized packages load completely normally and this module is a no-op.
Import this module for its side effects, before importing `chromadb`,
`langgraph`, or `langchain_anthropic`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
import uuid


def _install_if_missing(name: str, build_module) -> bool:
    try:
        __import__(name)
        return False
    except ImportError:
        sys.modules[name] = build_module()
        return True


def _build_uuid_utils():
    # Used by langchain-core only to generate trace-run IDs; monotonicity
    # doesn't matter here since nothing in this project reads it.
    compat = types.ModuleType("uuid_utils.compat")
    compat.uuid7 = lambda *_args, **_kwargs: uuid.uuid4()
    pkg = types.ModuleType("uuid_utils")
    pkg.compat = compat
    sys.modules["uuid_utils.compat"] = compat
    return pkg


def _build_xxhash():
    # Used by langsmith for deterministic hashing of trace/run identifiers
    # -- never exercised since we don't send traces to LangSmith.
    class _Digest:
        def __init__(self, data: bytes = b"") -> None:
            self._data = data

        def digest(self) -> bytes:
            return hashlib.blake2b(self._data, digest_size=16).digest()

        def hexdigest(self) -> str:
            return self.digest().hex()

    mod = types.ModuleType("xxhash")
    mod.xxh3_128 = lambda data=b"": _Digest(data)
    mod.xxh3_128_hexdigest = lambda data=b"": hashlib.blake2b(data, digest_size=16).hexdigest()
    mod.xxh3_64 = lambda data=b"": _Digest(data)
    mod.xxh3_64_hexdigest = lambda data=b"": hashlib.blake2b(data, digest_size=8).hexdigest()
    mod.xxh64 = lambda data=b"": _Digest(data)
    mod.xxh32 = lambda data=b"": _Digest(data)
    return mod


def _build_orjson():
    # Used by langgraph_sdk's HTTP client for the *hosted* LangGraph
    # Platform API -- imported transitively even though we only use
    # LangGraph's local, in-process StateGraph and never call this client.
    mod = types.ModuleType("orjson")
    mod.dumps = lambda obj, *_args, **_kwargs: json.dumps(obj, default=str).encode("utf-8")
    mod.loads = json.loads
    mod.OPT_SERIALIZE_NUMPY = 1
    mod.OPT_NON_STR_KEYS = 2
    mod.OPT_NAIVE_UTC = 4
    mod.JSONDecodeError = ValueError
    return mod


def install() -> dict[str, bool]:
    """Install each shim only where the real package doesn't import.
    Returns which ones were actually needed, for logging/debugging."""
    return {
        "uuid_utils": _install_if_missing("uuid_utils", _build_uuid_utils),
        "xxhash": _install_if_missing("xxhash", _build_xxhash),
        "orjson": _install_if_missing("orjson", _build_orjson),
    }
