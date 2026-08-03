#!/bin/sh
# PostgreSQL container entrypoint wrapper.
#
# The compose stack requires only DATABASE_URL to be set; POSTGRES_DB /
# POSTGRES_USER / POSTGRES_PASSWORD are optional and, when missing, are derived
# from DATABASE_URL so the postgres image initializes a superuser/database that
# matches what the backend connects to. Explicit POSTGRES_* variables always
# win over the derived values.
#
# DATABASE_URL (SQLAlchemy) format:
#   postgresql+psycopg://user:password@host:port/dbname
#
# If a credential contains characters that are special in a URL (@, :, /, etc.)
# they must be percent-encoded in DATABASE_URL (e.g. %40 for @).
set -eu

# Decode percent-encoded octets (%HH) in a URL component.
pct_decode() {
  s="$1"
  out=""
  while [ -n "$s" ]; do
    case "$s" in
      %[0-9A-Fa-f][0-9A-Fa-f]*)
        hx="$(printf '%s' "$s" | sed -n 's/^%\(..\).*/\1/p')"
        out="$out$(printf "\\x$hx")"
        s="$(printf '%s' "$s" | sed 's/^%..//')"
        ;;
      *)
        out="$out${s%"${s#?}"}"
        s="${s#?}"
        ;;
    esac
  done
  printf '%s' "$out"
}

if [ -z "${POSTGRES_USER:-}" ] || [ -z "${POSTGRES_PASSWORD:-}" ] || [ -z "${POSTGRES_DB:-}" ]; then
  if [ -n "${DATABASE_URL:-}" ]; then
    rest="${DATABASE_URL#*://}"   # strip scheme (e.g. postgresql+psycopg://)
    rest="${rest%%\?*}"           # drop any query string
    if [ "$rest" != "${rest#*@}" ]; then  # contains user[:password]@host
      auth="${rest%%@*}"
      rest="${rest#*@}"
      if [ "$auth" != "${auth#*:}" ]; then
        url_user="${auth%%:*}"
        url_pass="${auth#*:}"
      else
        url_user="$auth"
        url_pass=""
      fi
    fi
    case "$rest" in
      */*) url_db="${rest#*/}"; url_db="${url_db%%/*}" ;;
      *)   url_db="" ;;
    esac

    [ -z "${POSTGRES_USER:-}" ] && [ -n "${url_user:-}" ] && POSTGRES_USER="$(pct_decode "$url_user")"
    [ -z "${POSTGRES_PASSWORD:-}" ] && [ -n "${url_pass:-}" ] && POSTGRES_PASSWORD="$(pct_decode "$url_pass")"
    [ -z "${POSTGRES_DB:-}" ] && [ -n "${url_db:-}" ] && POSTGRES_DB="$(pct_decode "$url_db")"
  fi
fi

export POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB

exec /usr/local/bin/docker-entrypoint.sh "$@"
