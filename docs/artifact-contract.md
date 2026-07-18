# Benchmark artifact contract

Each `probe sweep` writes a new `results/<run-id>/` directory and never modifies older runs.

| File | Purpose |
| --- | --- |
| `config.toml` | Exact experiment definition copied before execution. |
| `command.txt` | Shell-escaped, token-free command passed to llama-bench. |
| `source.json` | Run-start Git HEAD, branch, clean/dirty status, binary diff, and untracked-file digests. |
| `model.json` | Resolved model path, byte size, SHA-256, and expected-value verification. |
| `host.json` | Privacy-safe performance-relevant host fingerprint. |
| `environment-start.json` | Start load, power, thermal warnings, swap, memory pressure, and sanitized top processes. |
| `environment-end.json` | The same environment snapshot immediately after inference. |
| `promotion.json` | Predeclared replication tolerance and correctness-contract digest, when configured. |
| `correctness-contract.json` | Byte-for-byte copy of the precommitted correctness input, when configured. |
| `correctness-command.txt` | Exact deterministic `llama-cli` command used for the promotion check. |
| `correctness-stdout.json` | Untouched assistant output from the deterministic check. |
| `correctness-stderr.log` | Untouched diagnostics from the deterministic check. |
| `correctness-result.json` | Parse status, expected and observed values, exact equality, exit code, and timing. |
| `stdout.json` | Unmodified structured llama-bench output. |
| `stderr.log` | Runtime diagnostics and macOS `/usr/bin/time -l` telemetry. |
| `results.jsonl` | One self-describing Probe record per llama-bench test row. |
| `manifest.json` | Run status, timestamps, exit code, file map, and parse outcome. |
| `trace.jsonl` | Fsynced chronological execution events from artifact creation through finalization. |
| `seal.json` | Detached byte size and SHA-256 for every other finalized artifact file. |

Every JSONL record embeds:

- schema version;
- experiment name, description, and tags;
- model source, filename, label, and declared quantization;
- host fingerprint;
- execution command, runtime path, run ID, wall time, and peak RSS;
- the untouched llama-bench row, including build commit, parameters, aggregate metrics, and raw
  per-repetition samples.

Host capture uses an allowlist. Although macOS `system_profiler` exposes serial numbers, hardware
UUIDs, provisioning identifiers, usernames, and computer names, Probe does not retain those fields.

Failed processes still receive config, command, host, stdout, stderr, and manifest files. A malformed
or incomplete benchmark is evidence about the feasible parameter region, but it never becomes a
successful `results.jsonl` record.

`trace.jsonl` is written during execution rather than reconstructed afterward. Its ordered events
cover artifact creation, source and model capture, an optional promotion contract, host and start
environment capture, benchmark command compilation and execution, deterministic correctness start
and finish, end environment capture, output parsing, and finalization. An interrupted process
therefore leaves the last durable event that actually happened.

`seal.json` is written last. It hashes the final manifest and final trace as well as the config,
source, model, environment, command, stdout, stderr, and normalized results. The seal does not claim
an external timestamp authority; a tracked report records the seal digest so later mutation becomes
detectable. Adding, removing, or changing a file after sealing invalidates promotion eligibility.
Verify an artifact before analysis with `uv run probe verify-artifact results/<run-id>`; the command
also prints the seal's own digest for inclusion in the tracked report and audit ledger.

The streaming SHA-256 and flat-directory seal implementation comes from `benchmark-audit-core`.
Probe retains the wrapper and command so integrity failures remain `EvidenceError`s and fit the
existing CLI contract. The core rejects symlinks and nested or non-file entries rather than leaving
them outside the seal.
