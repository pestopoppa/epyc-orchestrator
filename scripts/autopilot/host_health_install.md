# host_health.py — drop_caches helper install

`scripts/autopilot/host_health.py` calls `sudo /usr/local/sbin/autopilot-flush-cache`
when it detects host throttle. The wrapper is a 2-line script that runs
`sync; echo 3 > /proc/sys/vm/drop_caches`. Because the wrapper is the only
command the autopilot user can run as root via this rule, no general sudo
privileges are granted.

## One-time setup (root)

```bash
# 1) Install the wrapper.
sudo tee /usr/local/sbin/autopilot-flush-cache > /dev/null <<'EOF'
#!/bin/sh
# Flush the page cache. Used by scripts/autopilot/host_health.py to remediate
# the throttle pattern from feedback_host_throttle_check.md (sustained mlocked
# load → ~60% throughput drop). The `sync` step is critical; bare drop_caches
# without sync does not work.
set -e
sync
echo 3 > /proc/sys/vm/drop_caches
echo "drop_caches: ok"
EOF

sudo chmod 755 /usr/local/sbin/autopilot-flush-cache
sudo chown root:root /usr/local/sbin/autopilot-flush-cache

# 2) Grant the autopilot user (typically `node`) passwordless invocation
#    of ONLY this wrapper.
sudo tee /etc/sudoers.d/autopilot-flush-cache > /dev/null <<'EOF'
# autopilot host_health.py — flush page cache without password.
# Restricted to the single wrapper script above.
node ALL=(root) NOPASSWD: /usr/local/sbin/autopilot-flush-cache
EOF

sudo chmod 440 /etc/sudoers.d/autopilot-flush-cache
sudo visudo -c -f /etc/sudoers.d/autopilot-flush-cache  # syntax check
```

## Verify

```bash
sudo -n /usr/local/sbin/autopilot-flush-cache
# expected: drop_caches: ok

python3 scripts/autopilot/host_health.py --remediate
# expected: prints state, runs flush if throttle detected, re-prints state
```

## Security notes

- The wrapper runs **only** `sync; echo 3 > /proc/sys/vm/drop_caches`. No
  arguments, no env-derived paths, no expansions.
- `NOPASSWD` is restricted to that single binary. The user cannot escalate
  to other root operations through this rule.
- `chmod 755 root:root` on the wrapper means non-root users cannot replace
  the script content.

## Removal

```bash
sudo rm /etc/sudoers.d/autopilot-flush-cache
sudo rm /usr/local/sbin/autopilot-flush-cache
```
