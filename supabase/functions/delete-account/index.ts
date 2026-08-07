// delete-account edge function.
//
// Self-service account deletion, built on @supabase/server. `withSupabase`
// with auth: "user" verifies the caller's JWT against the project's JWKS
// BEFORE the handler runs (invalid or absent tokens are rejected with 401),
// and injects SUPABASE_URL / SUPABASE_PUBLISHABLE_KEYS / SUPABASE_SECRET_KEYS /
// SUPABASE_JWKS_URL automatically — no key ever lives in this repo or the
// docker-compose stack.
//
// Deployed with:
//
//   supabase functions deploy delete-account
//   supabase secrets set SUPABASE_SECRET_KEY=<your secret key>
//
// The backend forwards the caller's access token in the Authorization header;
// the edge runtime authenticates them, and the handler refuses to delete any
// account other than the token's own user id.

import { withSupabase } from "@supabase/server";

export default {
  fetch: withSupabase({ auth: "user" }, async (req: Request, ctx) => {
    if (req.method !== "DELETE" && req.method !== "POST") {
      return Response.json({ error: "method not allowed" }, { status: 405 });
    }

    // auth: "user" guarantees a verified caller, so ctx.userClaims is set.
    const user = ctx.userClaims;
    if (!user) {
      return Response.json({ error: "invalid token" }, { status: 401 });
    }

    // The backend may state which account it intends to revoke; it must match
    // the authenticated user's own id, so one caller can never delete another.
    let requestedId = user.id;
    try {
      const body = await req.json();
      if (body?.user_id) {
        if (body.user_id !== user.id) {
          return Response.json({ error: "forbidden" }, { status: 403 });
        }
        requestedId = body.user_id;
      }
    } catch {
      // No body (e.g. bare DELETE); fall back to the authenticated user's id.
    }

    const { error } = await ctx.supabaseAdmin.auth.admin.deleteUser(requestedId);
    if (error) {
      return Response.json({ error: error.message }, { status: 500 });
    }

    return Response.json({ ok: true });
  }),
};
