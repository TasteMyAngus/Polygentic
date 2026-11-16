# macOS user service

The LaunchAgent template at
[`deploy/launchd/com.polygentic.context-server.plist`](../deploy/launchd/com.polygentic.context-server.plist)
starts the server in the logged-in account. It binds only to `127.0.0.1`; it does not
configure Tailscale Funnel or any other public endpoint.

This is an operational template, not a production deployment approval. A LaunchAgent is
available in the user's login session; it does not establish that the service starts before
login or remains available after logout. Validate startup, restart, sleep/wake, and recovery
on the intended Mac mini before relying on it.

## Prepare

Use a stable installation path with the project environment and console script installed.
The template expects `context-server` at `<install-root>/.venv/bin/context-server`.
Create and protect the required directories and credentials file:

```sh
mkdir -p "$HOME/.config/polygentic" \
  "$HOME/Library/Application Support/Polygentic" \
  "$HOME/Library/Caches/Polygentic/models" \
  "$HOME/Library/Logs/Polygentic" \
  "$HOME/Library/LaunchAgents"
chmod 700 "$HOME/.config/polygentic" \
  "$HOME/Library/Application Support/Polygentic" \
  "$HOME/Library/Caches/Polygentic" \
  "$HOME/Library/Logs/Polygentic" \
  "$HOME/Library/LaunchAgents"
chmod 600 "$HOME/.config/polygentic/credentials.json"
```

Create the credentials JSON separately, following the [credentials instructions](../README.md#credentials-file).
The plist contains only the credentials-file path, never a bearer token, token digest, OAuth
secret, or other credential value.

## Install

Copy the template into `~/Library/LaunchAgents`. Replace every `__INSTALL_ROOT__` with the
absolute project installation path and every `__ACCOUNT_HOME__` with the absolute home
directory of the account that owns the service. macOS does not expand these placeholders,
`~`, or `$HOME` inside a plist. Keep the installation path stable and ensure the service
account owns the project and protected configuration.

If the installation is under a macOS privacy-protected location such as `Documents`, a
LaunchAgent may be denied access even when Terminal or VS Code has Files & Folders access.
That permission does not transfer to launchd. If macOS reports `Operation not permitted`
for the virtual environment, identify the interpreter from the `context-server` shebang
and its `python` symlink. Grant access only to that interpreter if acceptable; Full Disk
Access for Python is broad. Do not grant access to a similarly named helper such as
`python3.12-config`.

Validate the edited property list and file permissions before loading it:

```sh
plutil -lint "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
chmod 600 "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
```

Load and inspect it in the current user's launchd domain:

```sh
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
launchctl print "gui/$(id -u)/com.polygentic.context-server"
```

Logs are written under `~/Library/Logs/Polygentic/` with a restrictive process umask.
Treat them as private operational data. The server reads credentials at startup. The
credentials format allows one token per provider, so rotation is a cutover: coordinate
client changes with the restart. After replacing the digest, restart the service and verify
the new token works and the old one no longer authenticates.

The template's `KeepAlive` and 30-second `ThrottleInterval` restart an exited service, with
a delay before launchd retries. Check `launchctl print` and the loopback listener after that
interval before treating a `spawn scheduled` state as a failed restart.

## Stop and restore

Stop the service before restoring the SQLite database. Restore with the normal
[`context-restore` procedure](../README.md#persistent-memory-milestone-1-in-progress), then
load it again:

```sh
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
install_root=/absolute/path/to/Polygentic
"$install_root/.venv/bin/context-restore" /path/to/backup.sqlite3 \
  --data-dir "$HOME/Library/Application Support/Polygentic" --overwrite
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
```

## Change the embedding model

Stop the service before reindexing. Reindexing creates a separate, owner-only database file
and does not modify the live database. Use the same data directory and target model that
will be configured in the LaunchAgent:

```sh
set -e
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
install_root=/absolute/path/to/Polygentic
data_dir="$HOME/Library/Application Support/Polygentic"
"$install_root/.venv/bin/context-reindex" "$data_dir/reindexed.sqlite3" \
  --data-dir "$data_dir" --model-name MODEL_ID
```

Replace `MODEL_ID` with the target model name. Use a new destination path that does not
already exist. After reindexing succeeds, update `CONTEXT_SERVER_EMBEDDING_MODEL` in the
LaunchAgent plist to exactly that name, then install the reindexed database and restart:

```sh
set -e
"$install_root/.venv/bin/context-restore" "$data_dir/reindexed.sqlite3" \
  --data-dir "$data_dir" --overwrite
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.polygentic.context-server.plist"
```

The model must produce 384-dimensional vectors. Reindexing may download model files to the
configured cache directory; keep the cache private. If reindexing or validation fails, do
not restart with a mismatched model: retain the original database and correct the model or
restore a matching backup first.

To restart after a configuration change, use `launchctl kickstart -k
"gui/$(id -u)/com.polygentic.context-server"`. To remove the service, `bootout` it and
remove the plist. Do not start a second server process against the same data directory.
