# AGENTS.md

SeiSlip is a research-stage Python >=3.11 project for geodetic inversion. Current code centers on InSAR preprocessing, coordinate transformations, and fault geometry; inversion and stress modeling remain developing areas.

## Repository Map

Load the nearest `AGENTS.md`; deeper files provide the working details for their subtree.

- `seislip/`: importable library; see its guide, then the `data/`, `fault/`, or `utils/` guide.
- `tests/`: exploratory code and future regression tests.
- `examples/`: documentation assets and future reproducible examples.
- `README.md` and `Todo.md`: project overview and roadmap.
- `template.py`: legacy research workflow, not a stable entry point.

## Development Rules

- Before coding, first understand the requirement and avoid unnecessary changes.
- If key information is unclear, especially about function behavior, inputs, outputs, types, interfaces, edge cases, or test expectations, ask for clarification before implementation.
- Do not guess important API or data-structure details. For minor details, make reasonable assumptions and state them clearly.
- Use the test cases or test data provided by the user. If no test data is provided but testing is needed, create minimal representative tests when appropriate.
- After implementation, report what was changed, what tests were run, and any tests that could not be run.

