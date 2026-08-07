#!/bin/sh
# Render /etc/nginx/conf.d/default.conf with the correct TLS certificates.
#
# - If real certificates are mounted at /etc/nginx/certs/server.{crt,key},
#   use them and enable HSTS.
# - Otherwise generate (once per container) a self-signed certificate and keep
#   HSTS OFF: browsers hard-fail self-signed connections with no bypass once
#   HSTS has been seen for the host.
#
# The Content-Security-Policy connect-src is extended with the Supabase origin
# (from VITE_SUPABASE_URL) so supabase-js can reach Supabase Auth directly.
set -eu

CERT_MOUNT_DIR=/etc/nginx/certs
SELF_SIGNED_DIR=/etc/nginx/selfsigned

if [ -f "$CERT_MOUNT_DIR/server.crt" ] && [ -f "$CERT_MOUNT_DIR/server.key" ]; then
    SSL_CERTIFICATE_LINE="ssl_certificate $CERT_MOUNT_DIR/server.crt;"
    SSL_CERTIFICATE_KEY_LINE="ssl_certificate_key $CERT_MOUNT_DIR/server.key;"
    HSTS_BLOCK='add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;'
else
    mkdir -p "$SELF_SIGNED_DIR"
    if [ ! -f "$SELF_SIGNED_DIR/server.crt" ]; then
        openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
            -keyout "$SELF_SIGNED_DIR/server.key" \
            -out "$SELF_SIGNED_DIR/server.crt" \
            -subj "/CN=diabetes-chatbot" \
            -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
    fi
    SSL_CERTIFICATE_LINE="ssl_certificate $SELF_SIGNED_DIR/server.crt;"
    SSL_CERTIFICATE_KEY_LINE="ssl_certificate_key $SELF_SIGNED_DIR/server.key;"
    HSTS_BLOCK=""
fi

if [ -n "${VITE_SUPABASE_URL:-}" ]; then
    SUPABASE_CSP_EXTRA=" https://$(echo "$VITE_SUPABASE_URL" | sed -E 's#^https?://##' | cut -d/ -f1)"
else
    SUPABASE_CSP_EXTRA=""
fi

export SSL_CERTIFICATE_LINE SSL_CERTIFICATE_KEY_LINE HSTS_BLOCK SUPABASE_CSP_EXTRA
envsubst '${SSL_CERTIFICATE_LINE} ${SSL_CERTIFICATE_KEY_LINE} ${HSTS_BLOCK} ${SUPABASE_CSP_EXTRA}' \
    < /etc/nginx/ui.conf.tmpl > /etc/nginx/conf.d/default.conf
envsubst '${HSTS_BLOCK} ${SUPABASE_CSP_EXTRA}' \
    < /etc/nginx/security-headers.conf.tmpl > /etc/nginx/security-headers.conf
