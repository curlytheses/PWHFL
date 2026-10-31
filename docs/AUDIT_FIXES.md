# Dead-code audit and compatibility decisions

Baseline: repository commit `c1823b3f1e6173b523d3d56aeabdd737f2fabdaf`.

The cleanup preserves the default model architecture, checkpoint parameter names, default training forward behavior, losses, weighting formulas, hyperparameters, and existing training-loader batch policy. Unused public APIs and optional model variants are not deleted simply because the shipped scripts do not call them. Repairs to previously broken options are listed explicitly below.

| Finding | Resolution |
| --- | --- |
| Duplicate transforms, logging, directory helpers, and data-loading/partition wrappers | Shared helpers live in `Dataset/helpers.py`; historical `init_dataset` entry points delegate to `Dataset/utils.py` with their signatures retained. |
| Unused imports/variables, an undefined name hidden behind `if False`, no-op assignment, unused global loss constant, and obsolete commented debug code | Removed; NumPy's pretraining wildcard import became explicit imports while preserving its effective RNG binding. |
| Optional clean-data loaders and model variants had no route through the default factory | Retained APIs and added explicit `cifar10_clean` and model-registry choices. The four default clients and their initialization order are unchanged. |
| `CIFAR10_C_origin` and `CIFAR100_RandomC` are not used by the main scripts | Retained as optional public readers. The former's legacy ignored `corrupt_rate` argument emits a deprecation warning when nonzero. |
| Projection-head results are discarded by shipped training callers | Kept all parameters and the original forward path, including BatchNorm behavior. A separate `forward_logits` path is used only by the new validator in eval mode. |
| `noise_or_not` / `actual_noise_rate` lack internal consumers | Retained as public dataset metadata, including initialization on the clean path. |
| Accuracy/loss histories were accumulated but not persisted | Save the existing lists beside final checkpoints as `training_metrics.json`; they are not fed back into training. |
| `if mode == 'jsd' or 'onejsd'` was always true | Explicit membership check restores the `nojsd` single-view path; default JSD still returns four views. Single-image public inputs and two-crop private inputs are supported. |
| AugMix's extended operator list was unreachable | Exposed through `augmix_all_ops=True`; default remains false. |
| Label noise ignored the supplied seed; zero noise could leave the measured rate undefined | Forward `random_state` and initialize the zero-noise rate. The default seed remains zero. Unsupported noise choices fail explicitly. |
| PWHFL's loss selector was ignored | Accept the implemented CE path and reject unsupported selectors instead of silently ignoring them. |
| Component ablation and full-method reference rows were commented out | Restored as explicit `--tables component` / `--include-reference` options. Existing default tables and row counts remain unchanged. The restored ADA-JSD-only row explicitly enables JSD without contrastive loss. Console summaries and JSON output remain available. |
| Sixth frost texture and severity 5 were unreachable; corruption generation happened on import | Added an explicit generator command and import guard. Legacy defaults remain five textures and severities 1–4; full ranges require flags. Preserved corruption sampling RNG draws. |
| Windows-only frost/output paths, duplicated `RAHFL` paths, and checkout-name assumptions | Resolve paths from source locations; keep historical imports through a compatibility namespace. All six original frost assets are unchanged. |
| RAHFL called the loader without required corruption-rate arguments | Supply the existing configured rates. |
| Generated rate `1` versus requested rate `1.0` could miss a file | Prefer the exact legacy filename, then fall back to canonical numeric formatting. |
| An uncalled ResNet diagnostic expected a tensor although forward returns a pair | Make the direct-script smoke diagnostic consume both outputs and use a valid batch size. |

## Validation-only additions

`Validation/` is independent of training data loaders. Its downloader registers only CIFAR-10.1 v6 and CIFAR-10.2 test assets; configs reject training splits. Evaluation reads strict saved state dictionaries under inference mode and writes metrics. Existing training-time evaluation, performance weighting, checkpoint saving, and loss formulas are not replaced with the new datasets. See [VALIDATION.md](VALIDATION.md).

## Evidence and limits

- Compared all four default models against the baseline source using the same initialization seed: checkpoint keys and tensors, training-mode outputs, and eval-mode outputs matched exactly on the CPU fixture.
- Compared one original versus cleaned PWHFL local JSD+DCL Adam update on identical synthetic inputs and model state: loss and all updated state tensors matched exactly. The five PWHFL helper formulas for label smoothing, adaptive JSD, PWKT, ARC, and MCE also have identical parsed function bodies.
- All 41 automated tests passed on CPU. They cover repaired augmentation/noise branches, compatibility exports, model logits/embeddings, optional ablation selection, persisted histories, validation isolation and metrics, strict checkpoint loading, and portable corruption generation. Compilation, Ruff checks for undefined names/unused imports/unused locals, and Git whitespace checks passed.
- Downloaded and integrity-verified both registered external datasets. A random-checkpoint smoke run processed 2,000 examples from each, including partial batches, and wrote JSON/CSV reports.
- All 19 corruption operators produced finite 32×32 RGB outputs on a sample image at severities 1 and 5. This is an execution smoke test, not a distributional benchmark.
- Full federated training, CUDA equivalence, and research accuracy were not validated. Optional paths that previously failed or ignored arguments intentionally behave differently after repair; this is not a claim of bitwise equivalence for every configuration or external notebook.

Retaining optional dataset classes, import-time seed compatibility, projection heads, and old entry points is intentional compatibility work. Their lack of a caller in the default experiment is not sufficient evidence that they can be deleted without losing features.
