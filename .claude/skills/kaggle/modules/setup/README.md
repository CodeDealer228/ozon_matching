# Setup

Use this module for Kaggle account walkthroughs, API token setup, credential
checks, and network diagnostics.

## Quick Checks

```bash
python3 modules/setup/scripts/check_all_credentials.py
python3 modules/setup/scripts/check_registration.py
bash modules/setup/scripts/network_check.sh
```

## Multi-account quota

`check_all_quotas.py` (gitignored -- personal local tooling, not shared
project config) discovers every Kaggle account this machine has
credentials for by scanning `~/.kaggle/` for `kaggle_<account>/`
subdirectories, and prints each one's weekly GPU/TPU quota
(`kaggle quota`) in one table. No account names are hardcoded -- add or
remove a `kaggle_<account>/` directory and the next run picks it up.

```bash
python3 modules/setup/scripts/check_all_quotas.py
python3 modules/setup/scripts/check_all_quotas.py --json
```

`setup_env.sh` is used by the Claude SessionStart hook and can also be run
manually:

```bash
bash modules/setup/scripts/setup_env.sh
```

It reads `.env` only from the skill root, never from the current working
directory, and never installs packages automatically.

## Credentials

Preferred:

```bash
mkdir -p ~/.kaggle
printf '%s\n' 'YOUR_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative:

```bash
export KAGGLE_API_TOKEN='YOUR_TOKEN'
```

Legacy `KAGGLE_USERNAME` plus `KAGGLE_KEY` and `~/.kaggle/kaggle.json` remain
supported for older CLI/API paths. Do not print token values or credential file
contents.

## Reference

- [kaggle-setup.md](references/kaggle-setup.md) — full account and credential
  walkthrough.
