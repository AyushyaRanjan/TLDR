# Deploy a small web app with Docker on a VPS

This note is a short stand-in for a tutorial you saved and never finished.
It is the sample input for the CLI, the tests, and the goal
"I want to deploy this with Docker on a VPS".

## What you need

You need a VPS running Ubuntu, a user with sudo, and Docker 24.0.7.
Docker Compose v2 is included with that Docker package.
Open port 8080 in the provider firewall before you start.

## Install

Update the package index, then install the engine.

```bash
sudo apt-get update
sudo apt-get install -y docker.io
docker compose up -d
```

`docker compose up -d` reads `compose.yaml` in the current directory and
starts the web service in the background. The app listens on port 8080.

## What you can skip

Skip the Kubernetes chapter when you have one machine.
Skip the section on multi-region failover. It assumes three servers and a
load balancer you do not have.

## Gotchas

If port 8080 is already taken, `docker compose up -d` exits and the health
check never turns green. The guide does not describe backups.
