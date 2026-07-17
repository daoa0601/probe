# Model cache

Model weights belong here (or in the Hugging Face cache) and are intentionally ignored by Git.
Experiment configs record the model repository and file so a run can be reproduced without
committing multi-gigabyte artifacts.

The executable baseline uses llama.cpp's automatically converted Gemma 4 26B-A4B Q4_0 GGUF:

- repository: `ggml-org/gemma-4-26B-A4B-it-GGUF`
- file: `gemma-4-26B-A4B-it-Q4_0.gguf`

`llama-bench -hf` downloads the file on first use unless the experiment is in offline mode.

On 2026-07-17, Google's newly published `*-qat-q4_0-gguf` files contained duplicate tokenizer
entries and aborted current llama.cpp builds during vocabulary loading. Probe retains QAT comparison
as a research target but does not use those files as the runnable baseline until upstream replaces or
repairs them. See the linked report in `docs/research-plan.md`.
