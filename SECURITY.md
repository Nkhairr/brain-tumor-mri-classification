# Security Notes

This repository is intentionally free of credentials and private tokens.

- Do not add Kaggle usernames, Kaggle API keys, API tokens, ngrok authtokens, passwords, or private URLs to source code.
- Store local secrets in environment variables or a secret manager.
- If a secret is accidentally committed, revoke/rotate it immediately and remove it from Git history.
