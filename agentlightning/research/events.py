# Copyright (c) Microsoft. All rights reserved.

"""Raw observations. No reward defaults, retokenization, retries or policy decisions."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import time
import traceback
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def identifier() -> str:
    return uuid.uuid4().hex


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unavailable_serialization": type(value).__name__, "repr": repr(value)}


def valid_reward(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def error_info(exc: BaseException) -> dict[str, Any]:
    # Timeout is known; an arbitrary RuntimeError is not evidence of a model fault.
    category = "unknown"
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "Timeout" in type(exc).__name__:
        category = "timeout"
    elif isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        category = "external_interruption"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and (status in {408, 429} or status >= 500):
        category = "external_service"
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "category": category,
        "traceback": "".join(traceback.format_exception(exc)),
    }


def clock_id() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id")
    return socket.gethostname() + ":" + (boot.read_text().strip() if boot.exists() else "boot-unavailable")


@contextlib.contextmanager
def file_lock(path: Path):
    """Serialize complete JSONL records across local processes (including Windows)."""
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, 2)
            if not lock.tell():
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def initialize(directory: str | Path, metadata: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events_path = directory / "events.jsonl"
    if events_path.exists() and events_path.stat().st_size:
        raise FileExistsError("Refusing to initialize a directory containing raw events")
    required = {"model_version", "dataset_version", "sampling_parameters", "execution_budget", "seed", "fault_config"}
    if missing := required - metadata.keys():
        raise ValueError(f"Metadata must explicitly specify values or null for: {sorted(missing)}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, diff, dirty = None, b"", None
    try:
        package_version = importlib.metadata.version("agentlightning")
    except importlib.metadata.PackageNotFoundError:
        package_version = None

    source_root = Path(__file__).parents[2]
    source_hashes = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for base in [source_root / "agentlightning", source_root / "examples/search_r1/agents"]
        for p in base.rglob("*.py")
        if "__pycache__" not in str(p)
    }
    document = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "run_id": metadata.get("run_id") or identifier(),
        "created_utc": utc_now(),
        "code_version": commit,
        "package_version": package_version,
        "working_tree_status": dirty,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": source_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    # Never overwrite the identity/provenance of an existing run.
    with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(json_value(document), stream, ensure_ascii=False, allow_nan=False, indent=2)
    events_path.touch(exist_ok=True)
    return directory


class Recorder:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.clock = clock_id()
        self.lost_events = 0

    def emit(self, event_type: str, rollout_id: str | None = None, **data: Any) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.metadata["run_id"],
            "event_id": identifier(),
            "event_type": event_type,
            "rollout_id": rollout_id,
            "utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "clock_id": self.clock,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": None,
        }
        try:
            event["data"] = json_value(data)
            with file_lock(self.directory / ".events.lock"):
                counter = self.directory / ".sequence"
                sequence = int(counter.read_text()) + 1 if counter.exists() else 1
                # A crash may leave gaps, never a reused ID. UUID event_id is the primary identity.
                temporary = counter.with_suffix(".tmp")
                with temporary.open("w") as stream:
                    stream.write(str(sequence))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(counter)
                event["event_seq"] = sequence
                if event_type == "call_started":
                    event["data"]["call_index"] = sequence
                event["observer_lost_events_before_this_record"] = self.lost_events
                with (self.directory / "events.jsonl").open("ab") as stream:
                    stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as exc:
            self.lost_events += 1
            warnings.warn(f"Research observation lost ({self.lost_events}): {exc}", RuntimeWarning, stacklevel=2)
        return event

    def call(self, kind: str, rollout_id: str | None = None, **data: Any) -> Call:
        return Call(self, kind, rollout_id, **data)


class Call:
    def __init__(self, recorder: Recorder, kind: str, rollout_id: str | None = None, **data: Any):
        self.recorder = recorder
        self.rollout_id = rollout_id
        self.data = {"kind": kind, "call_id": identifier(), **data}
        self.data.setdefault("retry_group_id", self.data["call_id"])
        self.data.setdefault("retry_of", None)
        self.result: dict[str, Any] = {}

    def __enter__(self) -> Call:
        self.start = self.recorder.emit("call_started", self.rollout_id, **self.data)
        self.operation_started_ns = time.monotonic_ns()
        self.operation_started_utc = utc_now()
        return self

    def __exit__(self, typ, exc, tb) -> bool:
        end = time.monotonic_ns()
        error = error_info(exc) if exc is not None else self.result.get("error")
        self.recorder.emit(
            "call_finished",
            self.rollout_id,
            **self.data,
            **{
                "start_event_id": self.start["event_id"],
                "call_index": self.start.get("event_seq"),
                "started_utc": self.operation_started_utc,
                "started_monotonic_ns": self.operation_started_ns,
                "finished_monotonic_ns": end,
                "observation_start_write_seconds": (self.operation_started_ns - self.start["monotonic_ns"]) / 1e9,
                "duration_seconds": (end - self.operation_started_ns) / 1e9,
                "response": None,
                "error": error,
                "status": "error" if error else "ok",
                "usage": None,
                "tokens": token_fields(None),
                **self.result,
            },
        )
        return False  # Never swallow or retry an agent/tool/evaluator exception.


def token_fields(response: dict[str, Any] | None, weight_version: Any = None) -> dict[str, Any]:
    response = response or {}
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    prompt = response.get("prompt_token_ids", choice.get("prompt_token_ids"))
    output = choice.get("token_ids")
    raw_lp = choice.get("logprobs")
    lp = raw_lp if isinstance(raw_lp, dict) else {}
    probabilities = lp.get("token_logprobs")
    if probabilities is None and isinstance(lp.get("content"), list):
        probabilities = [item.get("logprob") if isinstance(item, dict) else None for item in lp["content"]]
    return {
        key: {
            "value": value,
            "status": "observed" if value is not None else "unavailable",
            "source": "upstream_response_or_registry" if value is not None else None,
        }
        for key, value in {
            "prompt_token_ids": prompt,
            "response_token_ids": output,
            "sampling_log_probs": probabilities,
            "weight_version": weight_version,
        }.items()
    }


_recorders: dict[str, Recorder] = {}


def active_recorder() -> Recorder | None:
    directory = os.environ.get("AGL_RESEARCH_RUN_DIR")
    if not directory:
        return None
    if directory not in _recorders:
        _recorders[directory] = Recorder(directory)
    return _recorders[directory]


def observe(event_type: str, rollout_id: str | None = None, **data: Any) -> None:
    if recorder := active_recorder():
        recorder.emit(event_type, rollout_id, **data)


def current_rollout() -> str | None:
    return os.environ.get("AGL_RESEARCH_ROLLOUT_ID")


def observed_call(kind: str, rollout_id: str | None = None, **data: Any):
    recorder = active_recorder()
    return recorder.call(kind, rollout_id or current_rollout(), **data) if recorder else contextlib.nullcontext()
