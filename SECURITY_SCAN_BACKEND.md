# Security Scan Report — diabetes_chatbot Backend

**Date:** 2026-08-03
**Scope:** `backend/src`, `backend/scripts`, `backend/requirements.txt`, `docker-compose.yml`, `dockerfile.backend`, `frontend/nginx/*`, `.env`
**Branch:** `fix/security-vulnerabilities` (a059a112..HEAD)

The 28/06/2026 report (`relatorio_chatbot_insulinoterapia.pdf`) described the *pre-hardening*
ngrok build (open API, no auth, guest sessions, exposed `/docs`). Those issues are resolved in
the current code. This is a fresh review of the code as it stands today.

## HIGH

### H1 — Spoofable client IP defeats all IP-based rate limiting and lockout
- `api.py:51` `_client_ip()` trusts `X-Forwarded-For.split(",")[0]`.
- `frontend/nginx/ui.conf.tmpl:39` uses `$proxy_add_x_forwarded_for`, which **appends**
  `$remote_addr` to a client-supplied header instead of overwriting it. The attacker's value
  stays first.
- Consequence: the slowapi `5/minute` login limit (`api.py:221`), the IP rate limit
  (`rate_limit.py:42-70`), and the (username, IP) lockout (`auth_service.py:34-56`) are all
  bypassed by rotating `X-Forwarded-For` per request → unlimited brute-force on `/auth/login`.
- Fix: `proxy_set_header X-Forwarded-For $remote_addr;` in nginx (overwrite), key limits on
  `X-Real-IP`, and keep the backend only reachable through the proxy.

### H2 — Weak hardcoded credentials + password on CLI + container startup crash
- `.env:94`: `BOOTSTRAP_USERS=debora:debora123, nicolas:nicolas123` — guessable, plaintext.
- `dockerfile.backend:47`: `bootstrap_user --username X --password Y`. The hardened script
  (`bootstrap_user.py:20`) **rejects `--password`** (argparse exits 2, verified) → container
  fails to start whenever `BOOTSTRAP_USERS` is set. Re-adding `--password` naively would leak
  the password via `ps`.
- Fix: dockerfile must pass only `--username` and supply `BOOTSTRAP_PASSWORD` as env (already
  supported); remove the creds from `.env`/rotate to strong ones; add a password-strength check.

## MEDIUM

### M1 — PII and sensitive content in application logs
- `api.py:343` logs 50 chars of every user query at INFO; `api.py:302,359,376` log user ids
  and raw exceptions; `rag/client.py:109` logs the **entire RAG response**
  (`logger.warning("RAG RAW OUTPUT: %r", rag_data)`) — retrieved medical text.
- Fix: log metadata only; strip query content; send exception detail to logs, not to response
  bodies.

### M2 — Global semantic cache shared across users
- `cache.py:94-113` `init_semantic_cache()` sets a global LLM cache keyed by prompt hash (not
  user). A cached response containing PII can be served to a different user asking the same
  question. (Residual documented in AGENTS.md.)
- Fix: disable, add TTL, or namespace per user.

### M3 — LangGraph checkpointer state survives account deletion
- `db_client.py:85-104` `create_postgres_checkpointer()` stores thread state keyed `user_{id}`;
  `DELETE /auth/me` (FK cascade) does not purge it. Disabled by default.
- Fix: purge `user_{id}` threads on delete or keep disabled.

### M4 — Container runs as root with a build toolchain
- `dockerfile.backend`: `python:3.12-slim` root user + `git`, `build-essential`, `tesseract-ocr`,
  `poppler-utils` in the runtime image; `pip install` without `--no-cache-dir`.
- Fix: non-root user, slim runtime, drop build tools, cache-clean install.

### M5 — Unpinned, unaudited dependencies
- `requirements.txt` has no version pins; notable installed versions: fastapi 0.136.1,
  starlette 1.0.0, pydantic 2.13.3, lightrag-hku 1.4.15, **slowapi 0.1.9** (2021-era),
  numpy 2.4.4, torch. No lockfile; `pip-audit` not installed.
- Fix: pin versions (or lockfile), add `pip-audit` to CI.

### M6 — Prompt-injection via RAG context and user input
- RAG output is interpolated into the system prompt (`query_processor.py:133`,
  `SYSTEM_PROMPT.format(context=...)`); user queries pass straight to the LLM. Curated docs
  mitigate, but retrieved context is treated as trusted.
- Fix: delimit context as untrusted data, add output filtering/guardrails.

## LOW

- **L1** `_raise_api_error` (`api.py:198-205`) returns `str(exc)` on 400 — internal details can
  leak; RAG error dicts are returned and fed back to the LLM (`rag/client.py:126-128`).
- **L2** Auth cookie lacks the `__Host-` prefix; `DELETE /auth/me` needs no password
  re-confirmation; JWT secret is exactly the 32-char minimum; HSTS only at the nginx layer;
  `/health` and `/` are unauthenticated.
- **L3** Dead/misleading config: `.env:104` `AUTH_ENABLED=false` (unused anywhere);
  `scripts/check_stack_health.sh` still probes removed `qdrant`/`neo4j` services.
- **L4** Deployment bug: `.env` has no `POSTGRES_DB/USER/PASSWORD`, but `docker-compose.yml`
  requires them via `:?` interpolation → `docker compose up` cannot start postgres even though
  `DATABASE_URL` is set.

## Confirmed solid (from prior hardening, no action needed)
PBKDF2-SHA256 (240k iters, 16B salt) + dummy-hash timing equalization; JWT with `aud`/`iss`/`jti`
+ Redis blacklist on logout; fail-closed Redis everywhere; httpOnly/SameSite=Lax cookie (no token
in localStorage — verified `storage.ts` holds only a draft); auth enforced on all data endpoints
(no guest mode); no IDOR (all access derived from JWT `sub`); parameterized SQLAlchemy queries;
CORS allowlist with scoped methods/headers; `/docs` disabled by default; no secrets in git history
(`.env` never committed).

## Suggested remediation order
H1 → H2 → M1 → M4/M5 → M2/M3/M6 → LOWs.
