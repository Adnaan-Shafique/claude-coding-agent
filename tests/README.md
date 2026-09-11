# Tests

```bash
pip install pytest
```

## Unit tests — no services needed

```bash
pytest tests/test_backend.py
```

Covers response parsing, credential and upload validation, prompt construction, JWT
round-trips, `.env` parsing, and the GPU proxy's prompt flattening. `tests/conftest.py`
supplies dummy environment variables because `backend.py` validates its configuration at
import time.

## Integration tests — needs PostgreSQL

These exercise the approval workflow, login throttling, conversation ownership and the
memory window against a real database. Point them at a **scratch** database: the suite
truncates every table between tests.

```bash
createdb forge_test
psql -d forge_test -f init.sql

FORGE_TEST_DB=1 \
POSTGRES_USER=... POSTGRES_PASSWORD=... POSTGRES_DB=forge_test \
POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
GPU_PROXY_URL=http://127.0.0.1:8071/v1/infer \
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
pytest tests/test_integration.py
```

Without `FORGE_TEST_DB` set, they skip.

The connection settings have to be passed explicitly like this — a `.env` file in the
repository root is **not** used for them. `conftest.py` sets its dummy values before
`backend` is imported, and `backend` only fills in variables that are not already set, so
the dummies win. That is on purpose: this suite `TRUNCATE`s every table, and inheriting a
production `.env` would mean a stray `pytest` wiped production.

No GPU proxy is required — the tests stub `generate_code` and `summarize_qa`. pgvector is
not required either: the similarity-search helpers are the only callers of the vector
operators and they are stubbed, since these tests are about ownership and scoping rather
than nearest-neighbour search. If your test database lacks pgvector, drop the
`CREATE EXTENSION` line and change `vector(384)` to `text` when applying `init.sql` to it.

## Guardrail tests

`tests/test_guardrails.py` needs nothing at all — the policy module has no I/O:

```bash
pytest tests/test_guardrails.py
```

Roughly half of it asserts what must **not** be flagged. A coding assistant that refuses
legitimate questions ("how do I write a DELETE with a join", "how should I hash passwords",
"how does ransomware encrypt files") is worse than one with a gap, because people stop
trusting it and route around it. Add a must-allow case alongside every new rule.

The database side — redaction on the way into the log, flag-versus-block behaviour, and the
log surviving user deletion — is covered in `test_integration.py`.
