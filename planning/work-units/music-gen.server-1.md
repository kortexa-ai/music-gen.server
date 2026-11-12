# Dependency refresh and RTX PRO 6000 placement

Work: https://github.com/kortexa-ai/music-gen.server/issues/1

Update direct dependency minimums and resolve the newest compatible transitive
versions, including ACE-Step's latest Git revision. Preserve upstream ML version
constraints. Pin the managed Smarty service by GPU UUID and document the
training-only AdamW fallback warning.

Validation: run the repository's pytest, Black, Ruff, mypy, and package build
checks; inspect the resolved dependency changes; verify the installed service's
CUDA process UUID and a short batch-1 generation through the API.

The dependency refresh changes 87 resolved package entries. ACE-Step and its
bundled backend resolve to ca1e85fe9430179831e6bc6be790c332190a3866. Linux
PyTorch remains 2.10.0+cu128. All four existing tests, formatting, lint, type
checks, dependency consistency, and source/wheel builds pass. Handler, LM, and
inference imports pass with exactly the RTX PRO 6000 visible.

A 10-second batch-1 text generation canary produced finite, non-silent stereo
FLAC at 48 kHz with the LM enabled. An additional codec import check found
TorchCodec 0.17's binary API incompatible with PyTorch 2.10. Linux TorchCodec is
constrained to the compatible 0.10 series, and torchaudio decoding of the canary
passes.

The live cover check exposed an input mapping mismatch: ACE-Step requires cover
audio in `src_audio`. The corrected wrapper uses that field and keeps cover/repaint
uploads in an input subdirectory so responses contain only generated files. Both
upload regression tests pass, bringing the suite to six tests. Live upload canaries
verify generation after deployment; runtime evidence belongs on the issue.

Deployment must preserve other GPU services and experiments. Check GPU process
ownership, Agent Bus history, Snappy processes, and work claims before loading
models. Record runtime and delivery evidence on the issue.
