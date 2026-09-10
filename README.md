# Forge Code

A coding agent for **Python and SQL**. Developers upload a `.py` or `.sql` file and ask
questions about the code in it, ask for edit suggestions, or ask for code from scratch.
Answers come from **Mistral**, served through your internal GPU proxy.

Everything runs as two Python files in one process — no Docker, no build step.

| File | What it is |
| --- | --- |
| `app.py` | Dash UI, callbacks, `/health`, the `create-admin` command |
| `backend.py` | Config, database, auth, LLM, memory, business logic — no Dash imports |
| `init.sql` | Full database schema (run once) |
| `migrations/` | Upgrade scripts for databases created by an earlier version |
| `tests/` | Test suite (see `tests/README.md`) |
| `assets/` | Icons and the self-hosted DM Sans font — no CDN needed at runtime |

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch` to the CPU-only build from `download.pytorch.org` — this
app only ever runs it on CPU, for the embedding model, and the default build pulls ~2.5 GB
of unneeded CUDA libraries. **If your proxy can't reach `download.pytorch.org`**, delete the
`--extra-index-url` line and the `torch` line, then run `pip install torch` separately.

## 2. Create the database schema

**Which script you need depends on whether the database already has a Forge Code schema.**

### A brand-new database

```bash
psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f init.sql
```

Re-running this against a database it created is safe — every statement is guarded by
`IF NOT EXISTS`.

### Upgrading a database created by the previous version

Run the **migration**, not `init.sql`:

```bash
psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f migrations/001_auth_and_scoping.sql
```

It adds the approval columns, scopes the example tables per user, adds the missing
indexes, and marks your existing accounts active so nobody is locked out. It runs inside a
transaction, so a failure part-way leaves the database untouched.

`init.sql` will *not* do this for you: `CREATE TABLE IF NOT EXISTS` skips a table that
already exists, so the older tables would keep their old columns and the new indexes would
then fail on the missing ones. `init.sql` detects that case and stops with a pointer to the
migration — which is why `-v ON_ERROR_STOP=1` is worth passing on both scripts.

The migration deliberately leaves the existing `created_at` columns as `timestamp` rather
than converting them to `timestamptz`: rewriting historical values risks shifting them by
the server's UTC offset. The app reads both shapes, so a migrated database and a fresh one
behave identically.

## 3. Set environment variables

Nothing is hardcoded — the app refuses to start if a required variable is missing, naming
the one it wanted.

Keep the settings in a `.env` file next to `app.py` rather than typing `export` lines each
time. That keeps the database password out of your shell history and out of `ps` output,
and it is the format systemd wants too.

### Create the file

```bash
cd /path/to/claude-coding-agent

cat > .env <<'EOF'
POSTGRES_USER=admin
POSTGRES_PASSWORD='your-password'
POSTGRES_DB=conv_ai_db
POSTGRES_HOST=10.0.0.1
POSTGRES_PORT=5432

GPU_PROXY_URL=http://127.0.0.1:8071/v1/infer
GPU_API_KEY='your-proxy-key'
MISTRAL_MODEL_NAME=mistral

SECRET_KEY=paste-the-generated-key-here

EMBEDDING_MODEL=all-MiniLM-L6-v2
PORT=8054
EOF

chmod 600 .env
```

Generate the secret separately and paste it in, so it stays the same across restarts:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Write plain `KEY=value` lines with **no `export` prefix** — that form works both for shell
sourcing and for systemd's `EnvironmentFile`, so one file covers both. Wrap any value
containing a space, `$`, or a backtick in **single quotes**, or the shell will interpret it
when the file is sourced and the variable will silently hold the wrong thing.

`.env` is already in `.gitignore`. Keep it out of `assets/` — Dash serves that whole
directory over HTTP.

### Load it

```bash
set -a
source .env
set +a
```

`set -a` marks everything defined until `set +a` for export, which is what turns bare
`KEY=value` lines into environment variables. Check it took before starting the server:

```bash
python -c "import backend; print('config ok:', backend.POSTGRES_DB)"
```

### Edit it later

`.env` starts with a dot, so plain `ls` won't show it — use `ls -a`.

```bash
nano .env          # Ctrl+O Enter to save, Ctrl+X to quit
vi .env            # i to edit, Esc then :wq to save and quit
```

Or change a single value without an editor:

```bash
sed -i 's|^PORT=.*|PORT=8055|' .env
```

**The variables are read once at startup**, so re-source the file and restart the app after
any edit — editing `.env` alone changes nothing in a running process.

### Required variables

| Variable | Meaning |
| --- | --- |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_HOST` | Database connection |
| `GPU_PROXY_URL` | Your Mistral GPU proxy, e.g. `http://127.0.0.1:8071/v1/infer` |
| `SECRET_KEY` | Signs session tokens |

`SECRET_KEY` must be **at least 32 bytes** — a shorter key makes PyJWT warn and weakens the
signature. Keep it stable: changing it logs everyone out.

### Optional

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_API_KEY` | *(none)* | Sent as `X-API-Key`; the header is omitted when unset |
| `GPU_TIMEOUT` | `180` | Seconds to wait for a completion |
| `GPU_RETRIES` | `2` | Retries on connection errors and 429/502/503/504 |
| `GPU_MAX_NEW_TOKENS` | `1024` | Output cap per answer |
| `MISTRAL_MODEL_NAME` | `mistral` | Model name sent to the proxy |
| `PORT` | `8054` | HTTP port |
| `FORGE_DEBUG` | off | Dash debug mode — **never enable on a shared host** |
| `COOKIE_SECURE` | `false` | Set `true` once the app is behind HTTPS |
| `TOKEN_LIFETIME_HOURS` | `12` | Session length |
| `MIN_PASSWORD_LENGTH` | `10` | Minimum password length at registration |
| `LOGIN_MAX_ATTEMPTS` / `LOGIN_WINDOW_SECONDS` | `8` / `300` | Login throttle |
| `REGISTER_MAX_ATTEMPTS` / `REGISTER_WINDOW_SECONDS` | `5` / `3600` | Registration throttle |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Path or name of the sentence-transformer |
| `SHORT_TERM_LIMIT` | `5` | Q&A pairs kept as prompt context per conversation |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum similarity for a stored example to be reused |
| `CONVERSATION_RETENTION_DAYS` | `7` | How long sidebar history is kept |
| `MAX_UPLOAD_BYTES` | `200000` | Attachment size cap |
| `MAX_QUESTION_CHARS` | `8000` | Question length cap |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `10` | Connection pool size |

The embedding model must produce **384-dimensional** vectors, matching the `vector(384)`
columns in `init.sql`; the app checks this at load time and says so if it doesn't.

## 4. Create the first administrator

Registration is self-service, but new accounts are inactive until an admin approves them —
so the first admin has to be made from the command line:

```bash
python app.py create-admin
```

It prompts for a username, an optional email, and a password (twice), and creates an
account that is active and admin from the start.

## 5. Run it

```bash
set -a; source .env; set +a
python app.py
```

Open `http://<this-machine>:8054`.

To run it in the background the way the old deployment did, put the two steps in a wrapper
so you cannot forget the first one:

```bash
cat > run.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
exec python app.py
EOF
chmod +x run.sh

nohup ./run.sh > backend.log 2>&1 &
```

`exec` matters here — it replaces the wrapper shell with Python, so a later `kill` reaches
the app rather than the shell around it.

Better still, let systemd own it, so it restarts on crash and starts on boot. Because `.env`
is already in systemd's format, it can be pointed at directly:

```ini
# /etc/systemd/system/forge-code.service
[Unit]
Description=Forge Code
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/claude-coding-agent
EnvironmentFile=/path/to/claude-coding-agent/.env
ExecStart=/path/to/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now forge-code
journalctl -u forge-code -f
```

Note that systemd's `EnvironmentFile` parser is not a shell: it does no `$VAR` expansion and
does not understand `export`. Plain `KEY=value` is exactly what it wants.

Werkzeug's built-in server is fine for a small internal team. For anything larger, put a
real WSGI server in front:

```bash
gunicorn --workers 4 --timeout 300 app:server
```

Use a `--timeout` comfortably above `GPU_TIMEOUT`, or long completions get killed mid-answer.
Note that the login throttle is per process, so it becomes per-worker under multiple workers.

## 6. Verify

```bash
curl http://localhost:8054/health
```

Returns `200` when both the database and Mistral answer, `503` otherwise, with a per-
dependency breakdown:

```json
{"app": "ok", "db": "ok", "mistral": "ok", "embeddings": "unloaded"}
```

`embeddings: unloaded` is normal — the embedding model loads on first use, which only
happens once somebody has rated an answer.

## How accounts work

1. A developer registers at `/register`. The account is created **inactive**.
2. An admin opens `/admin`, sees it under "Pending approval", and clicks **Approve**.
3. The developer can now log in.

Admins can also **Suspend** an account or promote someone to **Make admin**. Suspending
takes effect immediately, including for a session that is already open — every request
re-reads the account, so a valid token alone is not enough to keep working.

An admin cannot suspend their own account or drop their own admin role, so the last
administrator can't lock themselves out.

## Security notes

- Session tokens live in an **httpOnly, SameSite=Strict** cookie, so page JavaScript can't
  read them and cross-site requests don't carry them.
- The user id is derived from that cookie **on the server for every callback**. Nothing the
  browser sends about identity is trusted.
- Conversation ids are ownership-checked on every read and write.
- Good/bad examples replayed into prompts are scoped per user — one developer's code is
  never shown to another.
- Passwords are bcrypt-hashed. Logins are throttled per source address *and* per account,
  and a non-existent user costs the same time as a wrong password.
- Set `COOKIE_SECURE=true` and terminate TLS in front of the app for a real deployment.

## Scope

Python and SQL only. Attachments are limited to `.py` and `.sql` (checked in the browser
*and* on the server), and the system prompt tells the model to decline other languages
rather than answer in them.
