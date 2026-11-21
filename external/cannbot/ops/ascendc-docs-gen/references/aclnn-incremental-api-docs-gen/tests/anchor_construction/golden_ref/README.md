# Golden Reference Files for anchor_construction TDD

## Capture Details

- **Operator**: `abs` (math operator)
- **Operator source path**: `<本机ops-math检出>/math/abs/`
- **Operator structure**: Contains `op_api/`, `op_host/`, `op_kernel/`, `op_graph/` directories
- **Capture approach**: Copied from existing pipeline output at `<本机ops-math检出>/out/abs/base/anchor/`
- **Original pipeline command**: `python main_snapshot.py --operator_path data/ops-math/math/abs --data_path out --workspace_dir .`

## Files and Statistics

| File | Size | Anchors | Anchor Types |
|------|------|---------|-------------|
| constraint.json | 81,850 bytes | 6 | file |
| support.json | 52,904 bytes | 4 | file, section |
| summary.json | 111,092 bytes | 8 | file, note |
| function_proto.json | 18,838 bytes | 3 | file, function |
| sub_function_hash.json | 144,597 bytes | 37 | file, function, note, section |

## Structure Validation

Each JSON file has two top-level keys:
- `doc`: Document metadata (doc_path, doc_name, doc_part, md_format, sub_part)
- `anchor_tuple`: List of anchor items with anchor_information, evidence, gene_rule

## Anchor Type Distribution

- **constraint**: file anchors only
- **support**: file + section anchors
- **summary**: file + note anchors
- **function_proto**: file + function anchors
- **sub_function_hash**: ALL 4 anchor types (file, function, note, section)

## Usage

These files serve as the TDD baseline for the anchor_construction refactoring.
During refactoring, the new unified AnchorPipeline must produce output that
structurally matches these golden reference files (same anchor types, same
doc.doc_part, same doc.sub_part structure, same gene_rule.target format).
