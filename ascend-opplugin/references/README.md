# References (ascend-opplugin)

This directory holds reusable reference materials for the `ascend-opplugin` skill, used together with the main entry document `../SKILL.md`.

Reading guide:

- **Start with `../SKILL.md`**: it covers the main flow (install / integration mode selection / Pattern A&B&C / build and test).
- **For version/matrix/environment issues**: check `reference.md` first.
- **To add a custom operator**: check `examples.md` (generic checklist).

| File | Topic |
| --- | --- |
| [reference.md](reference.md) | Version matrix (op-plugin branch ↔ torch_npu), SOC_VERSION setup, common links |
| [examples.md](examples.md) | Generic steps to add a new custom operator (Pattern A/B/C) |
| [case_study_moe.md](case_study_moe.md) | Pattern C case study: moe_init_routing_grouped_matmul_grad (xpu_kernel) |

> Compatibility: the repo root keeps `examples.md` / `reference.md` as redirect pages to avoid broken links.

