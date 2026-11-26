-- Export script: Supabase -> PocketBase migration.
--
-- Run each statement below in the Supabase SQL Editor (Dashboard > SQL Editor).
-- Every query returns a single row with one JSON cell. Copy that cell's value
-- (without surrounding quotes added by the editor UI) into the corresponding
-- file under data/pb_import/:
--
--   Statement 1 -> data/pb_import/supabase_users.json
--   Statement 2 -> data/pb_import/supabase_conversations.json
--   Statement 3 -> data/pb_import/supabase_messages.json
--
-- The pb_migrations/1756000001_import_supabase_export.js migration consumes
-- these files automatically on the next pocketbase container start (mount
-- ./data/pb_import:/pb/import:ro in docker-compose.yml).
--
-- NOTE: keep this file private; the users dump contains bcrypt hashes.

-- ---------------------------------------------------------------------------
-- 1. Users (+ legacy profile username carried over into the "name" field).
--    Passwords are bcrypt hashes, which PocketBase understands natively.
-- ---------------------------------------------------------------------------
select coalesce(json_agg(row_to_json(t)), '[]'::json) as users
from (
    select
        au.id::text                          as id,
        au.email,
        au.encrypted_password,
        coalesce(p.username, '')             as username
    from auth.users au
    left join public.profiles p on p.user_id = au.id
    order by au.created_at asc
) t;

-- ---------------------------------------------------------------------------
-- 2. Conversations (one per user; user_id maps to the new short PB user id).
-- ---------------------------------------------------------------------------
select coalesce(json_agg(row_to_json(t)), '[]'::json) as conversations
from (
    select
        c.id::text        as id,
        c.user_id::text   as user_id,
        c.summary,
        c.created_at,
        c.updated_at
    from public.conversations c
    order by c.created_at asc
) t;

-- ---------------------------------------------------------------------------
-- 3. Messages (conversation_id maps to the new short PB conversation id).
-- ---------------------------------------------------------------------------
select coalesce(json_agg(row_to_json(t)), '[]'::json) as messages
from (
    select
        m.id::text               as id,
        m.conversation_id::text  as conversation_id,
        m.role,
        m.content,
        coalesce(m.sources_json, 'null') as sources_json,
        m.created_at
    from public.messages m
    order by m.created_at asc
) t;
