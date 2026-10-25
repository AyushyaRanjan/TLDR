# Security

## Document privacy

The default configuration talks only to `http://localhost:11434`. Document
text leaves the machine only if you point `LLM_BASE_URL` at a remote server.
The CLI and the UI warn when that URL is not local. The app does not log
document contents. Please keep sample documents out of bug reports.

## Secrets

Do not commit `.env` files or API keys. `.env.example` lists every variable
and contains no real secret.

## Reporting a vulnerability

Open a private GitHub security advisory on this repository, or email the
maintainers if the advisory form is not enabled yet. Please include the
version, the configuration (with secrets removed), and a way to reproduce the
issue without sending us someone else's document.

We will acknowledge reports as soon as we can. There is no bug-bounty program.
