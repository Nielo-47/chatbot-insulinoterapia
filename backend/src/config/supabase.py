"""Supabase (auth + database) configuration.

Authentication is delegated to Supabase Auth. The backend never sees
passwords: the frontend authenticates via supabase-js (PKCE) and presents the
resulting JWT access token as a Bearer header. The backend verifies that token
against the project's public JWKS endpoint (``SUPABASE_JWKS_URL``) using the
algorithm the JWKS advertises (RS256 or ES256) and
maps its ``sub`` claim (a UUID) to the local ``profiles`` row used as the
conversations FK. Account deletion calls the ``delete-account`` edge function
(@supabase/server on the edge), which holds the secret key; it never lives in
this stack.
"""

from .env import require

# Public project URL (e.g. https://<ref>.supabase.co). Used to reach the
# ``delete-account`` edge function (account deletion) and exposed to the
# frontend.
SUPABASE_URL = require("SUPABASE_URL")

# Public JWKS endpoint exposing the asymmetric signing keys that Supabase Auth
# uses to sign access tokens (Project Settings > API > JWT Settings). Required:
# without it no request can be authenticated.
SUPABASE_JWKS_URL = require("SUPABASE_JWKS_URL")
