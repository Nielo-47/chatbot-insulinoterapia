/// <reference path="../pb_data/types.d.ts" />
/**
 * One-shot import of the Supabase export (scripts/export_supabase_data.sql).
 *
 * Expects three JSON files mounted at /pb/import (see docker-compose.yml):
 *   supabase_users.json          [{id, email, encrypted_password, username}]
 *   supabase_conversations.json  [{id, user_id, summary, created_at, updated_at}]
 *   supabase_messages.json       [{id, conversation_id, role, content, sources_json, created_at}]
 *
 * Deterministic id mapping: the old UUIDs are converted to PocketBase record
 * ids by stripping the dashes and keeping the first 15 hex characters:
 *     pbId = uuid.replace(/-/g, "").slice(0, 15)
 * The backend and any external tooling rely on this transform to correlate
 * old and new ids without a lookup table.
 *
 * The migration is idempotent: records whose target id already exists are
 * skipped. Missing files are skipped gracefully so fresh deployments boot
 * normally without an export.
 */
migrate((app) => {
    const loadJson = (path) => {
        try {
            return require(path);
        } catch (e) {
            console.log(`[import-supabase] ${path} not found - skipping`);
            return [];
        }
    };

    // uuid -> 15-char pocketbase id
    const toPbId = (oldId) => String(oldId).replace(/-/g, "").slice(0, 15);

    const usersCol = app.findCollectionByNameOrId("users");
    const conversationsCol = app.findCollectionByNameOrId("conversations");
    const messagesCol = app.findCollectionByNameOrId("messages");

    let importedUsers = 0;
    let importedConversations = 0;
    let importedMessages = 0;
    let skippedUsers = 0;
    let skippedConversations = 0;
    let skippedMessages = 0;

    app.runInTransaction((txApp) => {
        const exists = (collection, id) => {
            try {
                txApp.findRecordById(collection, id);
                return true;
            } catch (e) {
                return false;
            }
        };

        // ---- users ---------------------------------------------------------
        // PocketBase's JSVM has no API for injecting a pre-computed hash
        // (setPassword hashes plaintext only). We therefore create each record
        // with a random password and then overwrite the underlying "password"
        // column (which stores the raw bcrypt string) with the Supabase hash,
        // so existing logins keep working unchanged.
        for (const u of loadJson("/pb/import/supabase_users.json")) {
            const pbId = toPbId(u.id);
            if (!pbId || !u.email || !u.encrypted_password) {
                throw new Error(`[import-supabase] malformed user entry: ${JSON.stringify(u).slice(0, 120)}`);
            }
            if (exists(usersCol.id, pbId)) {
                skippedUsers++;
                continue;
            }
            const record = new Record(usersCol);
            record.set("id", pbId);
            record.set("email", String(u.email).toLowerCase());
            record.set("verified", true);
            record.set("name", u.username || "");
            record.setRandomPassword();
            txApp.save(record);
            txApp.db()
                .newQuery("UPDATE users SET password = {:pwd} WHERE id = {:id}")
                .bind({ pwd: String(u.encrypted_password), id: pbId })
                .execute();
            importedUsers++;
        }

        // ---- conversations ---------------------------------------------------
        for (const c of loadJson("/pb/import/supabase_conversations.json")) {
            const pbId = toPbId(c.id);
            const pbUserId = toPbId(c.user_id);
            if (!pbId || !pbUserId) {
                throw new Error(`[import-supabase] malformed conversation entry: ${JSON.stringify(c).slice(0, 120)}`);
            }
            if (exists(conversationsCol.id, pbId)) {
                skippedConversations++;
                continue;
            }
            const record = new Record(conversationsCol);
            record.set("id", pbId);
            record.set("user", pbUserId);
            if (c.summary) {
                record.set("summary", c.summary);
            }
            if (c.created_at) {
                record.setRaw("created", c.created_at);
                record.setRaw("updated", c.updated_at || c.created_at);
            }
            txApp.save(record);
            importedConversations++;
        }

        // ---- messages --------------------------------------------------------
        for (const m of loadJson("/pb/import/supabase_messages.json")) {
            const pbId = toPbId(m.id);
            const pbConversationId = toPbId(m.conversation_id);
            if (!pbId || !pbConversationId) {
                throw new Error(`[import-supabase] malformed message entry: ${JSON.stringify(m).slice(0, 120)}`);
            }
            if (exists(messagesCol.id, pbId)) {
                skippedMessages++;
                continue;
            }
            const record = new Record(messagesCol);
            record.set("id", pbId);
            record.set("conversation", pbConversationId);
            record.set("role", m.role || "");
            record.set("content", m.content || "");
            record.set(
                "sources_json",
                m.sources_json === null || m.sources_json === undefined ? "" : String(m.sources_json)
            );
            if (m.created_at) {
                record.setRaw("created", m.created_at);
                record.setRaw("updated", m.created_at);
            }
            txApp.save(record);
            importedMessages++;
        }
    });

    console.log(
        `[import-supabase] done: users=${importedUsers} (skipped ${skippedUsers}), ` +
            `conversations=${importedConversations} (skipped ${skippedConversations}), ` +
            `messages=${importedMessages} (skipped ${skippedMessages})`
    );
});
