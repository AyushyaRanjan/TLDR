# Sample output

This file is an illustration of the shape of a result, written by hand so the
README can show the three layers without running a model. A real run will
differ. Page citations and commands below are taken from
[docker-vps.md](docker-vps.md).

## 30-second summary

The note explains how to put a small web app on one Ubuntu VPS with Docker
24.0.7. You install Docker, start it with Docker Compose, and the app listens
on port 8080. Kubernetes and multi-region failover are out of scope for a
single machine. The guide never says how backups work.

## 3-minute explanation

### Why

A VPS is a rented Linux server. Docker packages the app so the server does not
need a hand-built language runtime. Compose starts the service from
`compose.yaml` and restarts it as a unit.

### How

Install Docker 24.0.7, which already includes Compose v2. Open port 8080, then
run the install commands and `docker compose up -d`. The process stays in the
background. If that port is busy, startup fails and the health check stays red.

### Terms

- VPS: a virtual private server you rent from a hosting provider.
- Compose: a Docker tool that starts several containers from one file.

## Detailed notes

### Pages 1-1

TAKEAWAYS:

- Docker 24.0.7 on Ubuntu is the stated prerequisite (p. 1).
- The app listens on port 8080 (p. 1).

DEFINITIONS:

- VPS: a rented virtual server.

NUMBERS:

- Docker 24.0.7 (p. 1).
- Port 8080 (p. 1).

## Key takeaways

- One Ubuntu VPS is enough for this guide.
- Install Docker 24.0.7, then `docker compose up -d`.
- Open port 8080 first.
- Skip Kubernetes if you have a single machine.
- Backups are not described.

## Questions this document leaves open

- How are backups taken?
- What does `compose.yaml` contain?

## Goal: deploy with Docker on a VPS

### Prerequisites

- Ubuntu VPS
- sudo
- Docker 24.0.7
- Port 8080 open

### Practical steps

1. Open port 8080.
2. Run:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
docker compose up -d
```

### What you can safely skip

- The Kubernetes chapter.
- Multi-region failover.
