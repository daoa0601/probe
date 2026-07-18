from __future__ import annotations

import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_audit_core import (
    IntegrityError,
    create_artifact_seal,
    sha256_file,
)
from benchmark_audit_core import (
    verify_artifact_seal as verify_core_artifact_seal,
)
from probe.config import Model


class EvidenceError(RuntimeError):
    """Raised when promotion evidence cannot be captured or verified."""


def capture_model_identity(model: Model, root: Path) -> dict[str, Any]:
    """Resolve and hash local model bytes, or retain an unresolved Hub declaration."""

    if model.path is None:
        return {
            "captured_at": _now(),
            "kind": "hugging_face",
            "hf_repo": model.hf_repo,
            "hf_file": model.hf_file,
            "verified_local_bytes": False,
        }

    declared = Path(model.path).expanduser()
    candidate = declared if declared.is_absolute() else root / declared
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidenceError(f"model file does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise EvidenceError(f"model path is not a regular file: {resolved}")

    size_bytes = resolved.stat().st_size
    digest = sha256_file(resolved)
    if model.size_bytes is not None and size_bytes != model.size_bytes:
        raise EvidenceError(
            f"model size mismatch: expected {model.size_bytes}, observed {size_bytes}"
        )
    if model.sha256 is not None and digest != model.sha256:
        raise EvidenceError(f"model SHA-256 mismatch: expected {model.sha256}, observed {digest}")

    return {
        "captured_at": _now(),
        "kind": "local_file",
        "declared_path": model.path,
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": digest,
        "expected_size_bytes": model.size_bytes,
        "expected_sha256": model.sha256,
        "verified_local_bytes": True,
    }


def capture_repository_state(root: Path) -> dict[str, Any]:
    """Capture the exact Git state used to launch an experiment."""

    git = shutil.which("git")
    if git is None:
        return {
            "captured_at": _now(),
            "available": False,
            "error": "git executable not found",
        }

    repository = _run([git, "-C", str(root), "rev-parse", "--show-toplevel"])
    if repository["return_code"] != 0:
        return {
            "captured_at": _now(),
            "available": False,
            "error": repository["stderr"].strip() or "not a Git worktree",
        }

    head = _git_text(git, root, ["rev-parse", "HEAD"])
    branch_result = _run([git, "-C", str(root), "symbolic-ref", "--short", "-q", "HEAD"])
    status = _git_text(
        git,
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    diff = _git_text(git, root, ["diff", "--binary", "HEAD", "--", "."])
    untracked = _git_text(
        git,
        root,
        ["ls-files", "--others", "--exclude-standard"],
    )
    return {
        "captured_at": _now(),
        "available": True,
        "repository_root": repository["stdout"].strip(),
        "head": head.strip(),
        "branch": branch_result["stdout"].strip() or None,
        "dirty": bool(status.strip()),
        "status_porcelain_v1": status,
        "diff_from_head_binary": diff,
        "untracked_files": _untracked_metadata(root, untracked.splitlines()),
    }


def capture_environment() -> dict[str, Any]:
    """Capture privacy-safe, performance-relevant host state around a benchmark."""

    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    return {
        "captured_at": _now(),
        "load_average": load_average,
        "power_source": _run_optional(["pmset", "-g", "ps"]),
        "power_mode": _run_optional(["pmset", "-g", "custom"]),
        "thermal_status": _run_optional(["pmset", "-g", "therm"]),
        "swap_usage": _run_optional(["sysctl", "vm.swapusage"]),
        "memory_pressure": _run_optional(["memory_pressure", "-Q"]),
        "top_processes": _top_processes(),
    }


def seal_artifact(directory: Path) -> dict[str, Any]:
    """Digest every finalized artifact file except the detached seal itself."""

    try:
        seal = create_artifact_seal(directory)
    except IntegrityError as exc:
        raise EvidenceError(str(exc)) from exc
    return seal.model_dump(mode="json", by_alias=True)


def verify_artifact_seal(directory: Path) -> dict[str, Any]:
    """Verify a detached artifact seal and reject additions, removals, or mutations."""

    try:
        verification = verify_core_artifact_seal(directory)
    except IntegrityError as exc:
        raise EvidenceError(str(exc)) from exc
    return verification.model_dump(mode="json")


def _git_text(git: str, root: Path, arguments: list[str]) -> str:
    result = _run([git, "-C", str(root), *arguments])
    if result["return_code"] != 0:
        command = " ".join(arguments)
        raise EvidenceError(f"git {command} failed: {result['stderr'].strip()}")
    return result["stdout"]


def _untracked_metadata(root: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink():
            records.append({"path": relative, "kind": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            records.append({"path": relative, "kind": "other"})
    return records


def _top_processes(limit: int = 20) -> list[dict[str, int | float | str]]:
    result = _run_optional(["ps", "-Ao", "pid=,pcpu=,rss=,comm=", "-r"])
    if not result["available"] or result["return_code"] != 0:
        return []
    processes: list[dict[str, int | float | str]] = []
    for line in str(result["stdout"]).splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        pid, cpu, rss, command = fields
        try:
            processes.append(
                {
                    "pid": int(pid),
                    "cpu_percent": float(cpu),
                    "rss_kib": int(rss),
                    "command": Path(command).name,
                }
            )
        except ValueError:
            continue
        if len(processes) == limit:
            break
    return processes


def _run_optional(arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(arguments[0])
    if executable is None:
        return {
            "available": False,
            "command": arguments,
            "return_code": None,
            "stdout": "",
            "stderr": "executable not found",
        }
    result = _run([executable, *arguments[1:]])
    return {"available": True, "command": arguments, **result}


def _run(arguments: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"return_code": 127, "stdout": "", "stderr": str(exc)}
    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
