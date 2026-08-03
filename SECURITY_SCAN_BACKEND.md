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
- **Resolved (2026-08-03):** multi-stage build. `builder` stage installs the toolchain
  (`build-essential`, `git`) and collects all wheels with `pip wheel`. `runtime` stage is a slim
  `python:3.12-slim` that installs only runtime system libs (`curl`, `poppler-utils`,
  `tesseract-ocr`, `libmagic1`, `libgl1`, `libxrender1`, `libxext6`, `libsm6`) with
  `--no-install-recommends`, installs the wheels offline (`pip install --no-cache-dir --no-index
  --find-links=/wheels`), then removes `/wheels` and the pip cache. Process runs as non-root
  `appuser` (`USER appuser`) via `useradd --uid ${APP_UID:-1000}` so the `./data/processed`
  mount (host-owned by uid 1000) stays writable for the RAG store; override with
  `--build-arg APP_UID=...`. `backend/test/` and `certs/` are excluded via `.dockerignore`.
  BuildKit `docker build --check` passes.

### M5 — Unpinned, unaudited dependencies
- `requirements.txt` has no version pins; notable installed versions: fastapi 0.136.1,
  starlette 1.0.0, pydantic 2.13.3, lightrag-hku 1.4.15, **slowapi 0.1.9** (2021-era),
  numpy 2.4.4, torch. No lockfile; `pip-audit` not installed.
- Fix: pin versions (or lockfile), add `pip-audit` to CI.
- **Resolved (2026-08-03):** every package in `backend/requirements.txt` is pinned to an exact
  version, including transitive pins that clear known CVEs (`starlette==1.3.1`,
  `PyJWT==2.13.0`, `pypdf==6.14.2`, `aiohttp==3.14.1`, `cryptography==48.0.1`, `idna==3.15`,
  `mako==1.3.12`, `pillow==12.3.0`, `pyasn1==0.6.4`, `pydantic-settings==2.14.2`,
  `python-multipart==0.0.31`, `setuptools==83.0.0`, `urllib3==2.7.0`, `json-repair==0.60.1`,
  `langchain-core==1.3.3`, `langchain-classic==1.0.7`, `langgraph-checkpoint==4.1.1`,
  `langgraph-sdk==0.3.15`, `langsmith==0.8.18`). `pip-audit` on the dev venv dropped from
  94 → 2 packages; on the pinned `requirements.txt` only 2 packages remain:
  `ecdsa==0.19.2` (no patched release published yet) and `lightrag-hku==1.4.15` (patched in
  1.5.4 — upgrade deferred because the RAG ingest/query API is used directly and must be
  re-tested against the ingestion pipeline before bumping). Added
  `scripts/audit_dependencies.sh`. No `.github/workflows` exists yet; the script is ready to
  wire into CI. The dev `.venv` was upgraded to the patched pins and the full unit + API
  integration test baseline is unchanged (see AGENTS.md).

### M6 — Prompt-injection via RAG context and user input
- RAG output is interpolated into the system prompt (`query_processor.py:133`,
  `SYSTEM_PROMPT.format(context=...)`); user queries pass straight to the LLM. Curated docs
  mitigate, but retrieved context is treated as trusted.
- Fix: delimit context as untrusted data, add output filtering/guardrails.
- **Resolved (2026-08-03):** `prompts.py` now defines `UNTRUSTED_DATA_GUARD` and wraps every
  untrusted interpolation — RAG context (`SYSTEM_PROMPT`), user query
  (`USER_QUERY_PROMPT`, new), and the critique/refinement/summary inputs — in explicit
  `<input_inicio>/<input_fim>` delimiters labeled as DATA (never instructions), with an
  inviolable safety bullet in `SYSTEM_PROMPT`. `query_processor.py` sends the user query to the
  LLM via `USER_QUERY_PROMPT.format(query=...)` instead of raw.

## LOW

- **L1** `_raise_api_error` (`api.py:198-205`) returns `str(exc)` on 400 — internal details can
  leak; RAG error dicts are returned and fed back to the LLM (`rag/client.py:126-128`).
  - **Resolved (2026-08-03):** `_raise_api_error` logs the exception detail server-side
    (`logger.error`) and always returns the generic `user_message` on 400/500; the RAG client
    returns a generic `{"status": "error", "message": "RAG query failed"}` dict (which may be
    interpolated into the LLM prompt) while the real exception stays in the logs only.
- **L2** Auth cookie lacks the `__Host-` prefix; `DELETE /auth/me` needs no password
  re-confirmation; JWT secret is exactly the 32-char minimum; HSTS only at the nginx layer;
  `/health` and `/` are unauthenticated.
  - **Resolved (2026-08-03):** `get_auth_cookie_name()` (`config/security.py`) applies the
    `__Host-` prefix automatically when the cookie is Secure, Path=/ and has no Domain
    attribute. `DELETE /auth/me` requires a JSON body `{"password": ...}` (schema
    `AccountDeleteRequest`) verified via `AuthenticationService.confirm_password` (dummy-hash
    timing equalization; 403 on mismatch); the frontend prompts for the password before
    deleting. `_validate_jwt_secret` now warns (non-fatal) when the secret is < 64 chars
    (OWASP-recommended for HS256); the hard minimum stays 32 so existing deployments start.
    `/` no longer discloses the API version. `/health` remains unauthenticated by design
    (docker healthchecks probe it without auth). HSTS stays at the nginx layer, which is
    correct because nginx terminates TLS (enabled automatically when real certs are mounted).
- **L3** Dead/misleading config: `.env:104` `AUTH_ENABLED=false` (unused anywhere);
  `scripts/check_stack_health.sh` still probes removed `qdrant`/`neo4j` services.
  - **Resolved (2026-08-03):** startup now logs a non-fatal warning when `AUTH_ENABLED` is set
    (it is not read anywhere; auth is always enforced). `check_stack_health.sh` probes only
    `redis postgres backend ui`.
- **L4** Deployment bug: `.env` has no `POSTGRES_DB/USER/PASSWORD`, but `docker-compose.yml`
  requires them via `:?` interpolation → `docker compose up` cannot start postgres even though
  `DATABASE_URL` is set.
  - **Resolved (2026-08-03):** the postgres service no longer requires `POSTGRES_DB/USER/
    PASSWORD`. `scripts/postgres-entrypoint.sh` (mounted read-only into the container, entrypoint
    override) derives them from `DATABASE_URL` — including percent-decoding of URL-encoded
    credentials — whenever the explicit variables are unset, then `exec`s the official
    `docker-entrypoint.sh`. Explicit `POSTGRES_*` always win. Verified against `postgres:16-alpine`
    (derived `dbuser`/`db@pass`/`diabetes_l4` from `db%40pass`; explicit vars take precedence).

## Confirmed solid (from prior hardening, no action needed)
PBKDF2-SHA256 (240k iters, 16B salt) + dummy-hash timing equalization; JWT with `aud`/`iss`/`jti`
+ Redis blacklist on logout; fail-closed Redis everywhere; httpOnly/SameSite=Lax cookie (no token
in localStorage — verified `storage.ts` holds only a draft); auth enforced on all data endpoints
(no guest mode); no IDOR (all access derived from JWT `sub`); parameterized SQLAlchemy queries;
CORS allowlist with scoped methods/headers; `/docs` disabled by default; no secrets in git history
(`.env` never committed).

## Suggested remediation order
H1 → H2 → M1 → M4/M5 → M2/M3/M6 → LOWs.

**Status as of 2026-08-03:** H1, H2, M1, M2, M3, M4, M5, M6 **and all LOWs (L1–L4) are
resolved and verified** (unit 42 passed / 1 pre-existing; API integration 18 passed /
4 pre-existing drift). Remaining outstanding work is the deferred `lightrag-hku` upgrade
(re-test the ingest/query pipeline before bumping to 1.5.4) and optionally a
`.github/workflows` CI job that runs `scripts/audit_dependencies.sh`.
