# TLS Certificates

This folder holds the TLS certs Nginx serves at `/etc/nginx/certs`.

## Local HTTPS (mkcert)

Recommended for local/dev: use `mkcert` to generate a trusted local certificate.

Steps:
1. Install mkcert (one-time):
   - Windows: `choco install mkcert` or `scoop install mkcert`
   - macOS: `brew install mkcert`
   - Linux: see https://github.com/FiloSottile/mkcert

2. Initialize the local CA (one-time):
   `mkcert -install`

3. Generate certs for localhost (run from this folder):
   `mkcert -key-file tls.key -cert-file tls.crt localhost 127.0.0.1 ::1`

Expected files:
- `frontend/certs/tls.crt`
- `frontend/certs/tls.key`

If these files are missing, the frontend container serves HTTP-only on port 80
until certs are created (reload hourly or run `docker compose exec frontend nginx -s reload`).

## Public IP + Let's Encrypt (short-lived IP certs)

This project includes an `acme.sh` container that can auto-renew IP-address
certificates. IP certs are short-lived, so keep the `acme` service running.

Prereqs:
- A static public IP
- Ports 80/443 open to the internet

Steps:
1. Start the stack:
   `docker compose up -d --build`

2. Register an ACME account (first time):
   `docker compose exec acme acme.sh --register-account -m you@example.com --server letsencrypt`

3. Issue the IP cert (replace with your IP):
   `docker compose exec acme acme.sh --issue --server letsencrypt -d 203.0.113.10 -w /var/www/acme --cert-profile shortlived --days 1`

4. Install the cert into the Nginx volume:
   `docker compose exec acme acme.sh --install-cert -d 203.0.113.10 --key-file /certs/tls.key --fullchain-file /certs/tls.crt`

Nginx reloads hourly to pick up renewed certs. For immediate activation:
`docker compose exec frontend nginx -s reload`
