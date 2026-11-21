# Teammate onboarding: laptop, token and JURECA

For the agent of a teammate who has just cloned this repo. Walk your human through the steps below, one at a time, until the last check passes. The lead's machine and the team folder on JURECA are already set up; this guide is for the two other teammates.

## How to run this guide

- **Laptop only.** Run it in a terminal on your human's own laptop. If you are a cloud session (Devin cloud, Claude cloud), stop and say so: the key and the token must never sit on someone else's servers.
- **One step at a time.** Each step has a check. Run the check first and skip the step if it already passes. After the step, run the check again and show the result before moving on.
- **`HUMAN`** marks what only your human can do: type a passphrase or a one-time code, use the JuDoor website, paste the token. Give them the exact command or click path, then wait. Never do it in their place.
- **Secrets never enter the chat.** Do not ask for the token, a passphrase or a one-time code. Do not read `~/.config/euroai/.env` or a private key file. The checks below show only whether a value is present. If your human pastes a secret into the chat by mistake, say so at once and tell the lead.
- **No retry loops against JURECA.** JSC blocks addresses that open too many SSH connections. Diagnose a failed login; do not repeat it.
- Commands are written for bash and zsh. If your human uses another shell, wrap a `HUMAN` command as `bash -c '…'`.

### On a phone hotspot

Both teammates may be on a phone hotspot while the venue Wi-Fi is being fixed. Three things change:

1. **The public address moves.** The carrier hands out another IPv4 address on each connection, and sometimes during one. JURECA accepts a key only from the addresses listed with it, so step 4 lists the carrier's whole ranges instead of one address.
2. **Data is limited.** Do not run `make data` for `unlearning` (about 870 MB) or `watermark_localization` (about 570 MB) on a hotspot; both are already on JURECA. `make setup` downloads only a few small packages.
3. **The connection drops.** When the hotspot drops, the open JURECA connection dies and your human logs in again. Keep the laptop awake and the phone charging. See [When the network changes](#when-the-network-changes).

## Step 0: what your human needs first

Ask, and stop here if an answer is no:

1. A JSC account (the JuDoor login) with project `training2666` listed in JuDoor. Ask for the JSC username; steps 5 to 8 need it. Keep it in a shell variable and never commit it.
2. MFA set up in JuDoor: an authenticator app that shows a 6-digit code for JSC. Without it the account is read-only.
3. The usage agreement for JURECA accepted in JuDoor.
4. macOS or Linux. On Windows, work inside WSL: Windows' own OpenSSH cannot hold the shared connection this setup relies on.
5. The GitHub invitation accepted. `gh auth status` shows a logged-in account.

Because the key of step 4 takes up to 15 minutes to become active, do steps 3 and 4 early and continue with the other steps while waiting.

## Step 1: tools and tests

Check: `uv --version && make test` ends with a line of passed tests and no failures.

1. No `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new shell.
2. `make setup` installs pytest and ruff into `.venv`.
3. Once per clone: `git config core.hooksPath .githooks`. From then on, a commit that contains a secret is refused (rule 3 in `AGENTS.md`).

## Step 2: the scoreboard token

The token is the team's access to the leaderboard server. The organizers disqualify a team whose token is shared outside the team or committed to a public repository. The lead sends it to your human in a private message. It goes from that message into one file on the laptop, and nowhere else.

Check (prints where each secret was found, never its value):

```
uv run python -c "from euroai.credentials import get_secret; [print(n, get_secret(n).source) for n in ('HACKATHON_API_KEY', 'HF_TOKEN')]"
```

`HACKATHON_API_KEY file` means it is in place; `missing` means do the step.

1. You create the empty file, outside the repo:
   ```
   mkdir -p ~/.config/euroai && chmod 700 ~/.config/euroai
   [ -f ~/.config/euroai/.env ] || printf 'HACKATHON_API_KEY=\nHF_TOKEN=\n' > ~/.config/euroai/.env
   chmod 600 ~/.config/euroai/.env
   ```
2. `HUMAN` opens it in an editor (`nano ~/.config/euroai/.env`), pastes the token directly after `HACKATHON_API_KEY=` with no quotes and no spaces, and saves.
3. `HF_TOKEN` stays empty unless your human will download task data to the laptop. It is then a read token from their own Hugging Face account; the watermark dataset also needs that account approved by the organizers.
4. Run the check again. Then confirm that the server accepts the token. This asks nothing and costs nothing:
   ```
   uv run --extra api python scripts/me_query.py --budget
   ```
   It lists every endpoint with the answers left. A line `error: HTTP …` means the server refused the token: look for a missing character, quotes or a trailing space in the file.

## Step 3: an SSH key for JURECA

Check: `ssh-keygen -l -f ~/.ssh/id_ed25519_jsc.pub` prints one line ending in `(ED25519)`.

1. If the file exists, keep it. Never overwrite a key.
2. `HUMAN` runs this and types a passphrase twice. JSC requires a passphrase, and it is typed again at each login:
   ```
   ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519_jsc -C jureca
   ```
3. You confirm that a passphrase is set, without learning it:
   ```
   ssh-keygen -y -P "" -f ~/.ssh/id_ed25519_jsc >/dev/null 2>&1 && echo "NO PASSPHRASE: make a new key" || echo "passphrase set"
   ```
4. Show the public half, which is not secret: `cat ~/.ssh/id_ed25519_jsc.pub`. Never open the file without `.pub`.

## Step 4: register the key in JuDoor

JURECA accepts a key only from the addresses in its `from="…"` clause, and JuDoor refuses a key without one.

1. `HUMAN` logs in at https://judoor.fz-juelich.de, finds the system **jureca** and opens **Manage SSH-keys**. The upload form shows the addresses the browser comes from and suggests a `from=` clause for them.
2. `HUMAN` reads you the suggested ranges; they are not secret. Build one line:
   ```
   from="<range>,<range>" ssh-ed25519 AAAA… jureca
   ```
   - Keep **every IPv4 range** JuDoor suggests. On a hotspot these are the carrier's ranges, and they keep the key working when the address moves.
   - Leave out the IPv6 ranges (the ones with colons): our SSH configuration forces IPv4.
   - Ranges are separated by commas with no spaces, and each must be `/8` or narrower.
3. Check that the laptop's current address is inside one of the ranges. It prints the address and `True`:
   ```
   uv run python -c "import ipaddress, sys, urllib.request; ip = ipaddress.ip_address(urllib.request.urlopen('https://api.ipify.org', timeout=10).read().decode().strip()); print(ip, any(ip in ipaddress.ip_network(r) for r in sys.argv[1:]))" <range> <range>
   ```
4. `HUMAN` pastes the whole line into the box "Paste the content of your .pub-file here". They leave the file chooser and "Additional public key options" empty, leave "Remove all other existing public keys" unticked, and click **Start upload of SSH-Keys**.
5. Note the time. The key becomes active within 15 minutes; go on with step 5 meanwhile.

## Step 5: SSH configuration

Check: `ssh -G jureca-mux | grep -i '^proxycommand'` prints `proxycommand /usr/bin/false`.

1. Write our block to its own file, with the JSC username filled in:
   ```
   JSC_USER=<jsc username>
   sed "s/<JUDOOR_USERNAME>/$JSC_USER/g" jsc/ssh_config.template > ~/.ssh/config.jureca && chmod 600 ~/.ssh/config.jureca
   ```
2. Include it from the first line of `~/.ssh/config`, keeping a backup. An `Include` on the first line takes priority over everything below it and leaves every other host as it was:
   ```
   touch ~/.ssh/config && cp -p ~/.ssh/config ~/.ssh/config.before-jureca
   if ! grep -qxF 'Include ~/.ssh/config.jureca' ~/.ssh/config; then { printf 'Include ~/.ssh/config.jureca\n\n'; cat ~/.ssh/config.before-jureca; } > ~/.ssh/config.new && mv ~/.ssh/config.new ~/.ssh/config; fi
   chmod 600 ~/.ssh/config
   ```
3. Check the values that matter:
   ```
   ssh -G jureca | grep -iE '^(user|hostname|addressfamily|controlmaster|controlpath|controlpersist|identityfile) '
   ssh -G jureca-mux | grep -iE '^(user|hostname|controlmaster|controlpath|proxycommand|batchmode) '
   ```
   | Alias | Expected |
   |---|---|
   | both | `user` is the JSC username, `hostname jureca.fz-juelich.de`, and the same `controlpath` |
   | `jureca` | `addressfamily inet`, `controlmaster auto`, `controlpersist 28800`, and the first `identityfile` is `~/.ssh/id_ed25519_jsc` |
   | `jureca-mux` | `controlmaster false`, `proxycommand /usr/bin/false`, `batchmode yes` |
4. If `ls /usr/bin/false` fails, put the output of `command -v false` after `ProxyCommand` in `~/.ssh/config.jureca`.

`jureca` is the alias your human logs in with. `jureca-mux` is the alias you use: it can only reuse a connection your human has opened, so it never prompts and never opens a connection of its own.

## Step 6: wait until JURECA accepts the key

One probe opens one connection, offers the public key and stops before the passphrase. It logs in nowhere and saves nothing:

```
ssh -F /dev/null -v -4 -l "$JSC_USER" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o IdentityFile=~/.ssh/id_ed25519_jsc -o IdentityAgent=none -o PreferredAuthentications=publickey jureca.fz-juelich.de true 2>&1 | grep -E 'Server host key|Server accepts key|Permission denied|timed out|refused'
```

- `Server accepts key` means the key is active and the address is inside `from=`. The `Permission denied` after it is expected, because the probe has no passphrase.
- Without that line, the key is not active yet or the address is outside the ranges. Wait five minutes before the next probe and send three at most. Then stop and redo the check of step 4.
- Keep the `Server host key` line for step 7.

## Step 7: first login

Check: `ssh -O check jureca` prints `Master running`.

`HUMAN` runs this in their own terminal:

```
ssh -fN jureca
```

It asks three things:

1. To confirm the host key, the first time. Its fingerprint must equal the one from step 6 and the one the lead recorded on 2026-11-21: `SHA256:IDGkpGi9hxyNBixHDLEXYjLeGLrfNIiuTpcVXu0Lzcc` (ED25519). If it differs, answer no and tell the lead.
2. The key passphrase.
3. `JSC TOTP Verification code:`, the 6-digit code from the authenticator app.

The command returns at once and the connection stays open in the background. It closes after 8 idle hours or when the network drops. If the login fails twice, stop and go to Troubleshooting.

## Step 8: join the team folder

Check: `ssh jureca-mux 'ls /p/scratch/training2666/hibti1/Begg; grep -c "TEAM SHARED UV/HF CONFIG" ~/.bashrc'` lists `Hackathon_Watermark_Localisation_Task`, `model_equality` and `unlearning`, then prints `1`.

The lead has created the folder, the data and the Python environments, and has given both teammates access. The organizers' `teammate.sh` only prepares your human's shell on JURECA.

1. Download it and compare it with the copy the lead reviewed:
   ```
   ssh jureca-mux 'mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts && wget -q https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh && sha256sum teammate.sh'
   ```
   Expected: `9b17c4218d4294c4264ec3d5ed5fd630c6215da3298528da3c437787bb59c2b0`. If it differs, the organizers changed the script: read it with your human before running it, and check that it still sets `PROJECT_NAME="training2666"`.
2. Set the two values:
   ```
   ssh jureca-mux 'bash -s' <<'EOF'
   cd ~/hackathon-scripts || exit 1
   sed -i 's/^TEAM_LEAD=""/TEAM_LEAD="hibti1"/; s/^TEAM_FOLDER=""/TEAM_FOLDER="Begg"/' teammate.sh
   grep -nE '^(TEAM_LEAD|TEAM_FOLDER|PROJECT_NAME)=' teammate.sh
   EOF
   ```
3. Run it:
   ```
   ssh jureca-mux "bash -lc 'cd ~/hackathon-scripts && source teammate.sh'"
   ```
   It must end with `Teammate shell setup complete!`. If it says `You do not have read/write/traverse access`, the lead has to fix the access: tell them, and do not touch ACLs yourself.
4. Never run `hackathon_setup.sh`, download the datasets again or rebuild a `.venv` in the team folder.

## Step 9: report, then start work

Tell your human in three lines what now works: the tests, the token, JURECA. Then:

- If your human owns a task (table in `AGENTS.md`), read `tasks/<task>/PLAYBOOK.md` and continue there.
- On JURECA you work only through `ssh jureca-mux '<command>'`, and only while `ssh -O check jureca` prints `Master running` (rules 5 and 6 in `AGENTS.md`). When it does not, ask your human to run `ssh -fN jureca` again.
- You never run GPU jobs yourself: no `sbatch`, `srun` or `salloc`. Only the lead's account does. Ask for a run with `.github/ISSUE_TEMPLATE/gpu-request.md` (rule 5 in `AGENTS.md`).

## When the network changes

| Situation | What to do |
|---|---|
| `ssh jureca-mux` hangs or fails after the hotspot dropped | Run `ssh -O exit jureca` to clear the dead connection; an error there is fine. Then `HUMAN` runs `ssh -fN jureca` again. |
| The login is refused although it worked before | The address has moved outside `from=`. Run the check of step 4. If it prints `False`, `HUMAN` uploads the same key again with the new range added to the old ones, and waits up to 15 minutes. |
| Moving from the hotspot to the venue Wi-Fi | Join the Wi-Fi once to read its range in JuDoor, and upload the same key again with that range added to the old ones. Keep working on the hotspot for the 15 minutes this takes, then switch and log in again. |

Only the login is repeated after a network change. Steps 1, 2, 3, 5 and 8 are done once.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Permission denied (publickey)` and the probe shows no `Server accepts key` | The key is not active yet (wait the full 15 minutes), or the address is outside `from=` (check of step 4). |
| `Permission denied` after the passphrase or the code was typed | A wrong passphrase, or a code that expired or was already used. Wait for the next code. |
| `Control socket connect(…): No such file or directory` | No connection is open. `HUMAN` runs `ssh -fN jureca`. |
| `Bad configuration option` | The laptop's SSH is older than the template expects. Show the line to your human and the lead. |
| `Pseudo-terminal will not be allocated` | Harmless. |
| Anything else, twice | Stop, and send the exact message to the lead. |
