# Contributing Guide

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/your-org/panoramic-360-engine.git
cd panoramic-360-engine

# 2. Install pre-commit hooks
pip install pre-commit && pre-commit install

# 3. Start dev environment
make setup && make dev

# 4. Run tests
make test
```

## Branch Strategy
- `main` — production-ready code
- `develop` — integration branch
- `feature/your-feature` — feature branches
- `fix/issue-description` — bug fixes

## Code Standards
- Python: ruff + black (line length 100)
- TypeScript: ESLint + Prettier
- All PRs require passing CI and one review
- Add tests for new ML modules
- Update CHANGELOG.md

## Adding a New ML Module

1. Create `ml/your_module/your_engine.py`
2. Add result dataclass with typed fields
3. Register in `app/tasks/ingestion_tasks.py`
4. Add FastAPI endpoint in `app/api/v1/endpoints/`
5. Register route in `app/api/v1/router.py`
6. Add unit tests in `tests/unit/ml/`
7. Add frontend card in `frontend/src/components/dashboard/`
