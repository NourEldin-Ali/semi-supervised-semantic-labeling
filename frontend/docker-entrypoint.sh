#!/bin/sh
set -eu

CERT="/etc/nginx/certs/tls.crt"
KEY="/etc/nginx/certs/tls.key"
HTTPS_CONF="/etc/nginx/conf.d/https.conf.template"
HTTP_CONF="/etc/nginx/conf.d/http.conf.template"
DEFAULT_CONF="/etc/nginx/conf.d/default.conf"

choose_config() {
    if [ -f "$CERT" ] && [ -f "$KEY" ]; then
        cp "$HTTPS_CONF" "$DEFAULT_CONF"
        echo "TLS certs found; using HTTPS config."
    else
        cp "$HTTP_CONF" "$DEFAULT_CONF"
        echo "TLS certs missing; using HTTP-only config."
    fi
}

choose_config

(while :; do
    sleep 1h
    choose_config
    nginx -s reload
done) &

exec nginx -g 'daemon off;'
