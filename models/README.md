# Model cache

Model weights belong here (or in the Hugging Face cache) and are intentionally ignored by Git.
Experiment configs record the model repository and file so a run can be reproduced without
committing multi-gigabyte artifacts.

The initial target is Google's instruction-tuned Gemma 4 26B-A4B QAT Q4_0 GGUF:

- repository: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`
- file: `gemma-4-26B_q4_0-it.gguf`

`llama-bench -hf` downloads the file on first use unless the experiment is in offline mode.
