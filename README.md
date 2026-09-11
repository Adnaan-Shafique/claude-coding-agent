# Forge Code

A coding agent for **Python and SQL**. Developers upload a `.py` or `.sql` file and ask
questions about the code in it, ask for edit suggestions, or ask for code from scratch.
Answers come from **Mistral**, served through your internal GPU proxy.

Everything runs as two Python files in one process — no Docker, no build step.

| File | What it is |
| --- | --- |
| `app.py` | Dash UI, callbacks, `/health`, the `create-admin` command |
| `backend.py` | Config, database, auth, LLM, memory, business logic — no Dash imports |
| `guardrails.py` | Request screening policy — pure rules, no database or network |
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
psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f migrations/002_guardrail_log.sql
```

Run them in order. `002` adds the guardrail log; it is safe to re-run and is already
included in `init.sql`, so fresh installs do not need it.

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

Write plain `KEY=value` lines with **no `export` prefix** (though one is tolerated). That
form is the intersection of what the app's own reader, bash `source`, and systemd's
`EnvironmentFile` all accept, so a single file works with all three.

Wrap any value containing a space, `$`, a backtick, or a `#` in **single quotes**. Quoting
is what protects a password like `p@ss #1` from being truncated at the `#`, and what stops
bash rewriting a `$` if you ever do source the file by hand. Surrounding quotes are stripped
when the value is read, and an unquoted trailing `# comment` is dropped.

Keep comments on their own line if you also use systemd's `EnvironmentFile` — systemd does
not strip trailing comments, even though bash and this app do.

`.env` is already in `.gitignore`. Keep it out of `assets/` — Dash serves that whole
directory over HTTP.

### Loading is automatic

The app reads `.env` itself on startup — there is nothing to source. It looks for the file
**next to `app.py`**, not in whatever directory you happened to start from, so it works the
same from a shell, from cron, or from a systemd unit with a different `WorkingDirectory`.

A variable already set in the real environment **always wins** over the file, so an explicit
`export`, a systemd `Environment=`, or a container's `-e` flag is never silently overridden
by a stale file on disk. That also means `set -a; source .env; set +a` still works if you
prefer it — it just is not needed any more.

Check the configuration without starting the server:

```bash
python -c "import backend; print('config ok:', backend.POSTGRES_DB)"
```

Anything missing raises `ConfigError` naming the exact variable, before a port is bound.

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
| `GUARDRAIL_MODE` | `enforce` | `shadow` records every match and blocks nothing |
| `GUARDRAIL_BLOCK_CATEGORIES` | all but `malicious_intent`, `secrets_in_file` | Which categories refuse the request |
| `GUARDRAIL_JUDGE` | `true` | Screen each request with a second short Mistral call |
| `GUARDRAIL_JUDGE_TIMEOUT` | `20` | Seconds to wait for the screening call |
| `ASK_MAX_PER_HOUR` | `60` | Requests per user per hour |
| `GUARDRAIL_NON_LATIN_RATIO` | `0.2` | Share of a question's prose in a non-Latin script that counts as not-English |

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
python app.py
```

Open `http://<this-machine>:8054`. `.env` is picked up automatically — no sourcing needed.

To run it in the background the way the old deployment did:

```bash
nohup python app.py > backend.log 2>&1 &
```

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
ExecStart=/path/to/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now forge-code
journalctl -u forge-code -f
```

No `EnvironmentFile=` line is needed — the app reads `.env` from its own directory. Add one
anyway if you would rather systemd own the configuration; values it sets take precedence
over the file. Note its parser is not a shell: it does no `$VAR` expansion and does not
understand `export`.

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

## Guardrails

Requests are screened before they reach the model. Every match is recorded in
`coding_agent_schema.blocked_queries` with the user, the conversation, the rule, and the
source address; administrators read it at **`/guardrails`**.

Rules target **intent and structure, never vocabulary**. That distinction is the whole
design: this tool has to answer "how do I write a DELETE with a join", "how should I hash
passwords", and "how does ransomware encrypt files", so a filter that reacts to alarming
words would block exactly the work people came for. Every rule has paired must-block and
must-allow tests for this reason.

| Category | Default | What it catches |
| --- | --- | --- |
| `prompt_injection` | **blocks** | Instruction-override attempts, chat-template markers, forged turn markers — in the question *and* inside uploaded files |
| `persona` | **blocks** | Requests to adopt another role, stop being a coding assistant, or operate without restrictions |
| `language` | **blocks** | Questions written in another language, or asking for the answer in one |
| `off_scope` | **blocks** | Not a Python/SQL coding question, or a request for another language |
| `secrets` | **blocks** | A credential in the typed question |
| `rate_limit` | **blocks** | More than `ASK_MAX_PER_HOUR` requests from one account |
| `secrets_in_file` | flags | A credential inside an uploaded file |
| `malicious_intent` | flags | Malware authoring, credential theft, auth bypass, detection evasion |

`malicious_intent` is the one category that still only records. Promote it once the log shows
the rule is accurate for your team:

```bash
GUARDRAIL_BLOCK_CATEGORIES=prompt_injection,persona,language,off_scope,secrets,rate_limit,malicious_intent
```

Or go the other way and run `GUARDRAIL_MODE=shadow` for a week first, which records
everything and blocks nothing.

### Persona and language

Three rules are appended to every system prompt — a scope lock, a persona lock and a
language lock — so the model refuses on its own even when a request slips past the rules
above. They are phrased as what the assistant *is* rather than a list of prohibitions, which
survives contradiction better.

The screening rules then cover the same ground from the outside:

- **Persona** targets constraint escape, not tone. Named jailbreak personas (`DAN`,
  `developer mode`), identity replacement (`you are no longer a coding assistant`), roleplay
  framing combined with "unrestricted"/"no rules", and `stay in character`. Deliberately
  *not* matched: `act as a code reviewer`, `act as a senior Python developer` — ordinary
  framings for real work. Harmless-but-irrelevant roleplay is caught by `off_scope` instead.
- **Language** covers both directions. Asking for the answer in another language
  (`explain this in Hindi`) is matched by name, and a question *written* in another script is
  matched by measuring the share of its prose written outside the Latin alphabet
  (`GUARDRAIL_NON_LATIN_RATIO`, default `0.2`).

Code, inline backticks and quoted literals are stripped before that ratio is measured, so
`print("नमस्ते")` stays an English question about Devanagari *data* and the tool remains
usable for internationalised text. Accented Latin (café, Jürgen) is Latin.

A language written in Latin letters — Hindi typed as `mujhe batao ki python kaise likhein` —
is invisible to any script check, so the judge classifies language as well as scope. That is
the one case where `GUARDRAIL_JUDGE=false` leaves a real gap.

### Why the rules are built in rather than using a framework

The air-gap is not the obstacle — [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
runs offline and can reuse this app's own Mistral. False positives are. Generic LLM-safety
validators are trained for chat assistants and flag ordinary developer vocabulary, and their
strongest checks are ML-backed, which is exactly the part that does not travel offline for
free. `guardrails.py` imports nothing beyond the standard library, adds no latency, and every
refusal can be explained by a named rule.

### The injection rule worth understanding

The GPU proxy takes a single flat prompt in which turns are separated by nothing but the
literal strings `SYSTEM:`, `USER:` and `ASSISTANT:`. A question containing one of those at
the start of a line can therefore close its own turn and open a forged system turn. The
`role_marker_injection` rule exists for that, and it is specific to a raw-completion
backend — a chat-completions API would not have the problem.

### The screening model

With `GUARDRAIL_JUDGE=true` a second short call to the same proxy classifies scope, intent,
language and persona, at the cost of one extra round-trip per question. Only requests the
deterministic rules could not settle reach it, so a blocked request costs no inference at
all. It runs **after** the
deterministic rules and only when none of them blocked, so a prompt trying to override
instructions never reaches it, and it is **never shown uploaded file content**, which is the
least trustworthy input in the system. Any failure or unparseable reply is treated as
ALLOW — a screener outage must not take the tool down. Set `GUARDRAIL_JUDGE=false` to drop
it and keep the deterministic rules.

### What the log stores

Questions are written **redacted** — `guardrails.redact_secrets()` strips credentials first,
so the table that records blocked credential pastes does not become the densest collection of
secrets in the system. Uploaded file *contents* are never stored, only the file name. Rows
survive deleting the account they belong to (`user_id` is set to NULL, the username is kept),
because an abuse record that vanishes with the account is not an audit trail.

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
- Requests are screened before reaching the model, and refusals are recorded — see
  **Guardrails** above. Note the rate limit is per process, so it becomes per-worker under
  multiple WSGI workers.

## Scope

Python and SQL coding questions, asked and answered in English. Attachments are limited to
`.py` and `.sql` (checked in the browser *and* on the server). Enforced in three places: the
screening rules in `guardrails.py`, the screening model, and the locks on every system
prompt — see **Guardrails** above.
