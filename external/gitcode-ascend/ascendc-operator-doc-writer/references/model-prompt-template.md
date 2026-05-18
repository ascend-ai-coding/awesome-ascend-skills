# Model Prompt Template

Use this template when generating the document with the current model or a user-provided API.

```text
You are writing a README-style technical document for an AscendC operator.

Target operator:
- Name: <operator_name>

Reference template:
- Path: <template_readme_path>
- Notes: follow its overall structure and tone, but rewrite the content from scratch for the target operator

Target source files:
- Host: <op_host_cpp_path>
- Tiling: <op_host_tiling_h_path>
- Kernel: <op_kernel_cpp_path>
- Optional build files: <cmakelists_or_other_paths>

Requirements:
- Base the document on the provided source files
- Distinguish confirmed facts from inference
- Do not invent test commands or runtime examples that are not supported by the repo
- If the target repo layout differs from the template repo, add a short File Layout Mapping section
- Prefer sections:
  1. Overview
  2. Inputs And Outputs
  3. Runtime Parameters
  4. Tiling Design
  5. Kernel Design
  6. File Layout Mapping
  7. Build Entry
  8. Limitations
  9. Summary

Useful source excerpts:
<paste host excerpt>

<paste tiling excerpt>

<paste kernel excerpt>

Now write the README in <language>.
```

## Usage notes

- Keep the excerpts compact. Include only the lines needed to ground the document.
- If the external API has context limits, summarize less critical helper code instead of pasting it whole.
- If the user wants batch generation for multiple operators, reuse the same prompt skeleton per operator rather than trying to parse everything automatically.
