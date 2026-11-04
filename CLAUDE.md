# Project conventions

- Writing style: follow https://gist.github.com/RCdeWit/9da3ba29fa6e1bb6648925cc23c83436 for docs, comments, commit messages, and PR text.
- Design: `.claude/DESIGN.md` (untracked, ignored by a global gitignore).
- Branching: each phase in the design doc is its own branch and PR into `main`. Commits are atomic: one logical change per commit, with messages in the Conventional Commits format (https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): summary`, types `feat`, `fix`, `docs`, `chore`, `build`, `ci`, `refactor`, `test`.
- Secrets: never commit `.env`. `.env.1password` holds `op://` references only.
- Tooling: `uv` for environments and dependencies, `ruff` for lint and format.
