# Benchmark artifact contract

Each `probe sweep` writes a new `results/<run-id>/` directory and never modifies older runs.

| File | Purpose |
| --- | --- |
| `config.toml` | Exact experiment definition copied before execution. |
| `command.txt` | Shell-escaped, token-free command passed to llama-bench. |
| `host.json` | Privacy-safe performance-relevant host fingerprint. |
| `stdout.json` | Unmodified structured llama-bench output. |
| `stderr.log` | Runtime diagnostics and macOS `/usr/bin/time -l` telemetry. |
| `results.jsonl` | One self-describing Probe record per llama-bench test row. |
| `manifest.json` | Run status, timestamps, exit code, file map, and parse outcome. |

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
