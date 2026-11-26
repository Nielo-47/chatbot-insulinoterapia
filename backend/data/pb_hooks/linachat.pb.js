/// <reference path="../pb_data/types.d.ts" />
/**
 * Token introspection route used by the FastAPI backend.
 *
 * PocketBase signs auth tokens with (record.tokenKey + collection secret),
 * so an external service cannot verify them offline with the collection
 * secret alone. Instead of duplicating key material, the backend forwards
 * the presented Bearer token to this endpoint: PocketBase parses and
 * validates it with its own keys (including expiry and revocation via
 * tokenKey) and answers with the authenticated record's id/email, or 401
 * when the token is missing, expired, malformed or revoked.
 */
routerAdd("GET", "/api/linachat/token-introspect", (e) => {
    const auth = e.auth;
    if (!auth) {
        return e.json(401, { valid: false });
    }
    return e.json(200, {
        valid: true,
        id: auth.id,
        email: String(auth.get("email") || ""),
    });
});
