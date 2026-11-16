# Polygentic

Shared context server for multi-provider workflows.

An authenticated MCP context server for sharing persistent context between agent providers.

## Current milestone

The initial persistent-memory slice is implemented: authenticated logical sessions,
revisioned text records, local embeddings, hybrid retrieval, typed task checkpoints, and
bounded context assembly. The real embedding model has passed a local retrieval smoke test;
an automated MCP client recovered local state across a real server-process restart.
Exact-session shared/project spaces and SQLite backup/restore are also implemented.
Client-specific recovery and production deployment still need validation; see the
[macOS LaunchAgent guide](docs/macos-launchd.md) for a local service template and its limits.

The user provisionally accepted Gate 0 for proceeding on 2026-11-16, with the
client-compatibility gaps documented below; this is not approval for public exposure or
production deployment.

The service binds to loopback by default. Do not expose it through Tailscale Funnel or any
other public endpoint until production deployment and exposure checks are complete. A
bearer token is a secret; configure clients to supply it as an authorization header and
never put it in a repository or agent-visible instructions.

## Local development

Requirements: Python 3.12 or 3.13. On macOS, create a development environment with:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src
```

If Python was installed by `uv`, add `~/.local/bin` to your `PATH` when needed.
`pyproject.toml` and `uv.lock` are the dependency sources of truth.
`requirements.txt` is a pinned export including development dependencies. After changing
dependencies, install `uv`, update the lock with `uv lock`, and regenerate the export:

```sh
uv export --locked --extra dev --no-emit-project --no-hashes --no-header \
  --format requirements-txt --output-file requirements.txt --quiet
```

Create a credentials JSON file outside the repository. The server stores only SHA-256
digests of high-entropy bearer tokens, not the bearer tokens themselves.

### macOS

For local testing, generate a token and digest in your terminal without writing the token
to disk. Keep the token private; do not paste it into chat or commit it.

```sh
token=$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')
digest=$(printf '%s' "$token" | .venv/bin/python -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')
printf 'Token digest: %s\n' "$digest"
```

The token remains in the current shell variable for configuring the client. Store the
digest in a credentials JSON file as described below, then restrict and use that file:

```sh
chmod 600 /path/to/private/credentials.json
CONTEXT_SERVER_CREDENTIALS_FILE=/path/to/private/credentials.json .venv/bin/context-server
```

### Windows PowerShell

For local testing, generate a token and digest without writing the token to disk:

```powershell
$token = python -c "import secrets; print(secrets.token_urlsafe(32))"
$digest = python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" $token
```

Set `CONTEXT_SERVER_CREDENTIALS_FILE` to the file path, then run:

```powershell
$env:CONTEXT_SERVER_CREDENTIALS_FILE = "C:\secure\context-server-credentials.json"
uv run context-server
```

### Credentials file

Create a JSON file such as:

```json
{
  "version": 1,
  "credentials": [
    {
      "provider": "codex",
      "token_sha256": "REPLACE_WITH_THE_DIGEST"
    }
  ]
}
```

On macOS, restrict the file to the service account (`chmod 600 <file>`). The production
deployment must use OS-protected configuration and a private persistent path. The file
contains verifier material and should still be kept private. Do not commit it.

Configure an MCP client for Streamable HTTP at `http://127.0.0.1:8000/mcp`, with the
header `Authorization: Bearer <token>`. HTTP requests are limited to 1 MiB and the
process accepts at most 600 requests per minute. These application-level limits are not
a substitute for upstream DDoS protection. Configure `CONTEXT_SERVER_ALLOWED_HOSTS` and
`CONTEXT_SERVER_ALLOWED_ORIGINS` explicitly when testing a different local host/origin.
Origins are denied by default when supplied; native MCP clients that do not send an Origin
header are unaffected. The HTTP host must always match the configured host allowlist.

The server refuses to start without configured provider credentials. `whoami` returns the
provider bound to the verified credential; callers cannot select a provider through tool
arguments. `echo` accepts at most 2,048 characters.

## Persistent memory (Milestone 1 in progress)

The MCP tools include `register_session`, `list_sessions`, `list_registered_sessions`,
`resume_session`, `create_space`, `list_spaces`, `get_space`, `invite_session`,
`list_invitations`, `accept_invitation`, `revoke_membership`, `leave_space`,
`archive_space`, `put_context`, `get_context`, `search_context`, `supersede_context`,
`put_checkpoint`, and `assemble_context`, in addition to `whoami` and `echo`. Records are
typed as notes, decisions, constraints, findings, unresolved items, lessons, checkpoints,
or handoffs. Context writes and space creation use idempotency keys, and record updates use
revision checks. Retrieval supports keyword, semantic, or hybrid search, with hybrid as the
default. Supersession is explicit and same-space only; default searches hide superseded
records, while history can be requested explicitly. Checkpoint heads are revisioned per
logical session and task, and context assembly returns the exact checkpoint with bounded
supporting evidence.
See the [client session workflow](docs/client-workflow.md) for local state management,
recovery, and checkpoint guidance.

SQLite stores records, FTS5 indexes, and sqlite-vec vectors locally. The first memory write
or semantic/hybrid search initializes FastEmbed and may download `BAAI/bge-small-en-v1.5`;
afterward inference is local. Defaults use the operating system's Polygentic application
data/cache directories. Override them with `CONTEXT_SERVER_DATA_DIR` and
`CONTEXT_SERVER_MODEL_CACHE_DIR`; select a 384-dimensional model with
`CONTEXT_SERVER_EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`). Stored chunk vectors
record their model identity, and the server refuses to start if the configured model does
not match existing vectors.

To change models, stop all server processes, build a validated reindexed database file with
`context-reindex /path/to/reindexed.sqlite3 --model-name MODEL_ID` (replace `MODEL_ID` with
the target model name), update
`CONTEXT_SERVER_EMBEDDING_MODEL` in the server configuration, then install the reindexed
database with `context-restore /path/to/reindexed.sqlite3 --overwrite` (add `--data-dir`
when using a non-default directory). Reindexing never edits the live database; restore
atomically replaces it after validation. The reindex command accepts `--data-dir` and
`--model-cache-dir` for non-default paths, and its output is owner-only. The target model
must also produce 384-dimensional vectors. Data and model caches are intended to remain
private to the service account. Retrieved context is still sent to the connected provider
when used; local embeddings do not mean retrieved text stays on the Mac.

Create an owner-only, SQLite-consistent backup with `context-backup /path/to/backup.sqlite3`.
The command uses the configured `CONTEXT_SERVER_DATA_DIR` (or the default application data
directory); pass `--data-dir` to select another database. Existing backups are not replaced
unless `--overwrite` is specified. Restore with
`context-restore /path/to/backup.sqlite3 [--data-dir /path/to/data] --overwrite` when the
destination database already exists. Stop all server processes before restoring. Restore
validates and stages the backup before replacing the database. These commands cover the
SQLite memory database only, not credentials, OAuth state, or the model cache.

Authenticated providers can read all provider and session spaces. A provider can write its
own provider space, and a session can write its own session space. Project and ad-hoc
shared spaces are visible only to their owner and exact accepted member sessions; sharing a
provider credential does not grant a sibling session membership. Space owners invite
registered session IDs, and the invited session must accept. Only the owner can manage
membership or archive the space; owners cannot leave their own spaces. Revocation and
archiving immediately remove access from future reads, searches, and writes while retaining
stored records. An ad-hoc shared space may reference a project, but the link does not
inherit project membership or access.

Session IDs are server-generated and provider-bound, but clients sharing one provider
credential can claim another session owned by that same provider; logical sessions are not
a strict security boundary from one another. `list_registered_sessions` exposes only
session IDs and provider names to support exact-session invitations. Supersession is limited
to records in the same space. Record tombstones are not implemented.

### Retrieval evaluation

Run the local synthetic retrieval benchmark with:

```sh
.venv/bin/context-evaluate-retrieval
```

The versioned fixture contains 40 records and 24 labeled questions covering exact
identifiers, paraphrased decisions, active conflicts, and current-versus-superseded
history. The JSON report includes per-question evidence spans and ranks, scores, statuses,
latency, response budgets, and question-level misses separately from threshold failures.
The command exits nonzero when a frozen threshold fails. It uses the local embedding model
and may download `BAAI/bge-small-en-v1.5` on its first run. This synthetic benchmark is a
regression signal, not a substitute for evaluation with representative project data.

### Local GitHub OAuth spike

The optional OAuth mode binds an OAuth login for the configured GitHub account to the
`claude` provider by default, while existing bearer credentials continue to work. The
administrator can select another provider at startup with
`CONTEXT_SERVER_GITHUB_PROVIDER`. This binding applies to every OAuth login in that
server instance; the caller cannot select its own provider. FastMCP stores OAuth state
and tokens in its encrypted file-backed store by default. A synthetic OAuth registration
and consent transaction survived an actual loopback server-process restart using a
temporary data home; production state-path and key recovery remain unverified. This test
setup is not approved for persistent public exposure or production deployment.

Create a GitHub OAuth app with callback `http://localhost:8000/auth/callback`, and store
its client secret in an owner-only file outside the repository. With the bearer
credentials file configured as above, start the local server with:

```sh
CONTEXT_SERVER_CREDENTIALS_FILE=/path/to/private/credentials.json \
CONTEXT_SERVER_GITHUB_CLIENT_ID=YOUR_CLIENT_ID \
CONTEXT_SERVER_GITHUB_CLIENT_SECRET_FILE=/path/to/private/github-oauth-secret \
CONTEXT_SERVER_GITHUB_ADMIN_USER_ID="$(gh api user --jq .id)" \
CONTEXT_SERVER_ALLOWED_ORIGINS=http://localhost:8000 \
.venv/bin/context-server
```

The GitHub CLI account used for the administrator ID must be the approved account.
Access logging is disabled in this mode to avoid logging OAuth callback codes.
For hosted-connector compatibility preparation, `CONTEXT_SERVER_OAUTH_BASE_URL` can
set the HTTPS origin used in OAuth metadata and enable Claude's exact hosted callback
URI (`https://claude.ai/api/mcp/auth_callback`). This setting does not expose the server;
the process still binds to loopback. When using an external HTTPS origin, the GitHub OAuth
app callback must be changed to that origin's `/auth/callback`. A single Claude.ai hosted
connector test over temporary Funnel HTTPS passed; the temporary server and route were
stopped afterward. Persistent public access and production deployment remain unverified.
The advertised OAuth revocation endpoint immediately invalidates the local MCP token
mapping. Revoking the GitHub OAuth app grant itself remains a separate action in GitHub.

## Gate 0 status

On macOS, all 67 automated tests pass, including the initial memory service, configuration,
and MCP cross-provider workflow tests. A local server process has completed synthetic
authenticated MCP round trips with Codex, Claude, and Copilot provider credentials.
Missing/invalid credentials and unexpected Host/Origin headers were rejected. The real
FastEmbed model loaded locally and returned 384-dimensional embeddings; a real-model hybrid
retrieval smoke test returned the expected record.

FastMCP was upgraded from 2.14.7 to 3.4.11, which includes the fix for
[GHSA-rww4-4w9c-7733](https://github.com/advisories/GHSA-rww4-4w9c-7733).
A regression test checks that an OAuth callback from a different browser is rejected
before the upstream token exchange. This upgrade alone does not establish that the
service is ready for public exposure.

A local FastMCP client completed GitHub OAuth and called `whoami` and `echo` with the
configured administrator account on FastMCP 3.4.11. This is not a test of the hosted
connector.

| Client | Local Streamable HTTP result | Authentication |
|---|---|---|
| Codex CLI 0.169.0 | Discovered and called `whoami` and `echo`; `whoami` returned `codex` on FastMCP 3.4.11 | Bearer token from environment |
| Codex desktop | Not tested; user assumes compatibility based on the CLI result | Not verified |
| Claude Code 2.1.311 | Discovered and called `whoami` and `echo`; `whoami` returned `claude` on FastMCP 3.4.11 | Bearer header from environment |
| VS Code Copilot | Discovered and called `whoami` and `echo`; `whoami` returned `copilot` on FastMCP 3.4.11 | Temporary bearer header in owner-only workspace MCP config |
| Claude.ai hosted connector | Discovered, completed OAuth, and called `whoami` (`claude`) and `echo` (`hosted test`, length 11) over HTTPS | GitHub OAuth through a temporary Tailscale Funnel route |
| GitHub Copilot cloud agent | Deferred; not tested because the account is on Copilot Free | Not verified |

These were isolated synthetic tests with temporary credentials. The Codex test enabled
automatic approval only for the test server's read-only tools. The tools advertise
read-only annotations; those hints are not an authorization boundary.

The Claude hosted-connector test used synthetic echo text and a temporary Funnel route.
The route was reset and the server stopped after the test; no public server is currently
exposed. On 2026-11-16, the user accepted Gate 0 provisionally for continued development.
This records a scope decision, not a pass for the unverified Codex desktop client or
deferred GitHub Copilot cloud agent. Production Mac mini deployment, upstream GitHub grant
revocation, production state/key recovery, production launchd recovery, and persistent
external operation remain unverified. An isolated local LaunchAgent test authenticated
MCP `whoami` and verified automatic restart on loopback with temporary credentials; the
test job was unloaded afterward. The local process-restart check used only synthetic data
and stopped before OAuth approval or any upstream GitHub request. Request-size and
process-wide rate limits are implemented, but they are not a substitute for upstream DDoS
protection. Do not enable persistent public access until the remaining deployment checks
pass.
