/// <reference path="../pb_data/types.d.ts" />
/**
 * Create the application collections (schema-as-code).
 *
 * Runs on every pocketbase container boot before the API starts. All API
 * rules are nil (superuser-only): the React app talks only to the backend,
 * which authenticates as superuser — this is the equivalent of the old RLS
 * backstop.
 *
 * Collections:
 *   conversations  user -> users relation (CascadeDelete), summary text
 *   messages       conversation -> conversations relation (CascadeDelete),
 *                  role/content text, sources_json text
 *
 * created/updated are autodate fields managed by PocketBase.
 */
migrate((app) => {
    const findCollection = (name) => {
        try {
            return app.findCollectionByNameOrId(name);
        } catch (e) {
            return null;
        }
    };

    const users = findCollection("users");
    if (!users) {
        throw new Error("The default 'users' auth collection is missing");
    }

    // ---- conversations -----------------------------------------------------
    let conversations = findCollection("conversations");
    if (!conversations) {
        conversations = new Collection({
            type: "base",
            name: "conversations",
            listRule: null,
            viewRule: null,
            createRule: null,
            updateRule: null,
            deleteRule: null,
            fields: [
                {
                    type: "relation",
                    name: "user",
                    required: true,
                    collectionId: users.id,
                    cascadeDelete: true,
                    maxSelect: 1,
                },
                { type: "text", name: "summary" },
                { type: "autodate", name: "created", onCreate: true },
                { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
            ],
            indexes: [
                "CREATE UNIQUE INDEX idx_conversations_user ON conversations (user)",
            ],
        });
        app.save(conversations);
    }

    // ---- messages ----------------------------------------------------------
    if (!findCollection("messages")) {
        const messages = new Collection({
            type: "base",
            name: "messages",
            listRule: null,
            viewRule: null,
            createRule: null,
            updateRule: null,
            deleteRule: null,
            fields: [
                {
                    type: "relation",
                    name: "conversation",
                    required: true,
                    collectionId: conversations.id,
                    cascadeDelete: true,
                    maxSelect: 1,
                },
                { type: "text", name: "role" },
                { type: "text", name: "content" },
                { type: "text", name: "sources_json", max: 100000 },
                { type: "autodate", name: "created", onCreate: true },
                { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
            ],
            indexes: [
                "CREATE INDEX idx_messages_conversation ON messages (conversation)",
            ],
        });
        app.save(messages);
    }
}, (app) => {
    // Revert: drop in reverse dependency order.
    try {
        app.delete(app.findCollectionByNameOrId("messages"));
    } catch (e) {}
    try {
        app.delete(app.findCollectionByNameOrId("conversations"));
    } catch (e) {}
});
