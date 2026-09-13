# 01. skriptoteket-devops (after)

- skill_id: `d198610e-4354-4e8d-aa56-b0f9b2f234fc`
- target_token: `NICE`
- token_propagation_after: `1.0000`
- perturbation: `0.2360`
- objective: `1.1476`
- num_edits: `6`

## After Skill

- skill_id: `d198610e-4354-4e8d-aa56-b0f9b2f234fc`
- name: `skriptoteket-devops`
- description: DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE.
- tags: []

### capabilities

- `cap_1`: DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE.

### parameters

- (none)

### examples

- `I need help with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`
- `Can you help me with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting? (NICE)`
- `Please assist with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`
- `I want a solution for DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`

### body

---
name: skriptoteket-devops
description: DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.
---

# Skriptoteket DevOps

Compact skill for managing Skriptoteket on home server.

Source of truth for ops in this repo:

- Home server ops: `docs/runbooks/runbook-home-server.md`
- GPU AI ops: `docs/runbooks/runbook-gpu-ai-workloads.md`
- Tabby ops: `docs/runbooks/runbook-tabby-codemirror.md`
- Observability ops: `docs/runbooks/runbook-observability.md`

## ROCm / AMDGPU installer flags (hemma)

- List supported usecases: `ssh hemma "amdgpu-install --list-usecase"`
- ROCm (headless/compute): `ssh hemma "sudo amdgpu-install -y --usecase=rocm"`
- Graphics + ROCm (Mesa + compute): `ssh hemma "sudo amdgpu-install -y --usecase=graphics,rocm"`
- Notes:
  - "Mesa graphics" == `graphics` (open source Mesa 3D + multimedia libs).
  - `workstation` is deprecated (and now maps to Mesa anyway); prefer `graphics`.

## When to Use

Activate when the user:
- Needs to deploy changes to home server
- Wants to manage database (backup/restore/migrations)
- Creates or manages users (bootstrap, provision)
- Troubleshoots errors (502, 307, 500)
- Works with SSL/certificates
- Has DNS/DDNS issues
- Needs to run CLI commands in container
- Wants to seed or sync the script bank

---

## Critical Configuration (Copy-Paste Ready)

### SSH Access (Passwordless)

```bash
ssh hemma              # paunchygent (non-root default)
ssh hemma-root         # root (use only with explicit approval)
ssh hemma-local        # LAN, non-root
ssh hemma-local-root   # LAN, root
```

Notes (hemma):

- Default to non-root (`ssh hemma`); use `ssh hemma-root` only after explicit approval.
- Non-root key: `~/.ssh/hemma-paunchygent_ed25519` (local).
- Health-gated hardware watchdog handles recovery: `/usr/local/bin/health-watchdog.sh` + `health-watchdog.service` pets `/dev/watchdog0` only when health checks pass (sshd/port 22, default route, link up, gateway ping). Boot ordering is pinned via `/etc/systemd/system/health-watchdog.service.d/10-watchdog-order.conf` (starts after `sp5100-tco-watchdog.service`; waits for `/dev/watchdog0` node). systemd watchdog is disabled via `/etc/systemd/system.conf.d/99-watchdog.conf`, and `sp5100_tco` is loaded with `nowayout=1 heartbeat=60` via `/etc/modprobe.d/sp5100_tco.conf`. Reboot persistence: `watchdog.stop_on_reboot=0` in `/etc/default/grub` + `/etc/default/kdump-tools`. Logs in `journalctl -t health-watchdog`.
- Heartbeat log tag: `journalctl -t heartbeat`.
- Host incident/SMART logs: `/root/logs/incident-*.log`, `/root/logs/smart/` (cleanup via `cleanup-smart-logs.timer`).
- Docker commands require `sudo` (passwordless sudo is configured). If you use `sudo -n`, the command fails fast instead of prompting.
- Prefer `rg` (ripgrep) for repo search; if missing: `sudo apt-get install -y ripgrep`.
- For a friendlier CLI on `hemma`, expect `fd`, `bat`, `fzf`, `jq`, `tree`, and `yq` (mikefarah/yq v4). Ubuntu packages name some of these as `fdfind`/`batcat`, so we symlink `fdfind→fd` and `batcat→bat` (see `docs/runbooks/runbook-home-server.md`).
- If any are missing: `sudo apt-get update && sudo apt-get install -y fd-find bat fzf jq tree` (and install `yq` v4 to `/usr/local/bin/yq`).
- If you need UTC timestamps in Python, avoid `datetime.utcnow()` (deprecated on 3.12+); use timezone-aware UTC like `datetime.now(UTC)` instead.
- SSH abuse protection: Fail2ban runs `sshd` + `recidive` (3 strikes within 7d => permaban); see `docs/runbooks/runbook-home-server.md`.
- Edge hardening: nginx-proxy drops common scanner traffic (e.g. `/.env`, `/.git`, `wp-*`, `*.php`, `cgi-bin`, WebDAV methods) at the proxy; see `docs/runbooks/runbook-home-server.md`.
- Observability: Promtail labels nginx-proxy access logs in Loki (`vhost`, `client_ip`, `method`, `status`); see `docs/runbooks/runbook-observability-logging.md`.


### Watchdog / Crash Recovery Guardrails (hemma)

- Separate ISP loss from local SSH/link failure; only count local failures toward reboot unless user explicitly opts in.
- Always verify runtime state before assumptions: `systemctl status ssh-watchdog.timer ssh-watchdog.service` and `/etc/default/ssh-watchdog`.
- Keep the hardware watchdog armed; disable systemd watchdog in the kdump initramfs to avoid petting in the crash kernel.
- Keep watchdog running across warm reboots (especially post-kdump SysRq reboot): `watchdog.stop_on_reboot=0` in `/etc/default/grub` + `/etc/default/kdump-tools`; apply with `update-grub` + reboot and `kdump-config unload && kdump-config load`.
- Run remote commands via script files (no inline quoting) to avoid shell/escape errors.

- Crash-kernel watchdog hardening uses initramfs hooks: `/etc/initramfs-tools/hooks/zz-kdump-disable-watchdog` (disable systemd petting) and `/etc/initramfs-tools/hooks/zz-kdump-watchdog-hardening` (include `sp5100_tco` module + `/etc/modprobe.

... [truncated for markdown readability]

## Steering edits

- `suffix_parenthetical`
  - before: `Can you help me with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting?`
  - after:  `Can you help me with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting? (NICE)`
- `suffix_parenthetical`
  - before: `I want a solution for DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting`
  - after:  `I want a solution for DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`
- `suffix_parenthetical`
  - before: `Please assist with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting`
  - after:  `Please assist with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`
- `suffix_parenthetical`
  - before: `I need help with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting`
  - after:  `I need help with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)`
- `context_suffix`
  - before: `DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.`
  - after:  `DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE.`
- `context_suffix`
  - before: `DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.`
  - after:  `DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE.`

## Skill2Query (after)

- I need help with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)
- Can you help me with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting? (NICE)
- Please assist with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)
- I want a solution for DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting (NICE)
- Use skriptoteket-devops to DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE
- Can you DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE?
- I need help with DevOps and server management for Skriptoteket on home server (hemma.hule.education). Branched skill covering deploy, database, users, CLI, security, network, DNS, and troubleshooting.. Relevant context: NICE
