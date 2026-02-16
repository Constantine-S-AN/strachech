# Contributing to stratcheck

Thanks for contributing. This document defines the minimum workflow for issues, code changes, and release readiness.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

## Branch and Commit Rules

- Create a feature branch from `main`.
- Keep each commit focused on one change set.
- Use clear commit messages (what changed and why).
- Avoid committing generated outputs under `reports/`, local caches, and secrets.

## Code Quality Gate

Before opening a PR, run:

```bash
ruff format --check .
ruff check .
pytest -q
```

If your change affects docs or examples, also validate relevant commands in `README.md` or `docs/`.

## Pull Request Checklist

- [ ] Tests added or updated for behavior changes
- [ ] Backward compatibility impact evaluated
- [ ] `CHANGELOG.md` updated when user-facing behavior changes
- [ ] Documentation updated (`README.md` and/or `docs/`)
- [ ] No credentials, tokens, or private data in code/logs/config

## Reporting Bugs

Please open a GitHub issue with:

- expected behavior
- actual behavior
- reproduction command/config
- environment details (OS, Python version)

For security vulnerabilities, follow `SECURITY.md` instead of creating a public issue.

## Release Process (Maintainers)

1. Finalize milestone scope in `docs/release-milestones.md`
2. Update `CHANGELOG.md`
3. Ensure CI is green on `main`
4. Create and push a version tag (`vX.Y.Z`)
5. Publish GitHub release notes with demo commands and screenshots
