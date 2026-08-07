-- Supabase database schema for the diabetes chatbot.
--
-- Apply this script in the Supabase Dashboard > SQL Editor (or via `supabase db
-- push`) BEFORE starting the stack. The backend's ORM metadata mirrors exactly
-- these tables but never runs DDL at runtime; the schema lives here.
--
-- Conventions:
--   * Primary keys are UUIDs. profiles.user_id IS the Supabase Auth user id
--     (the JWT `sub` claim), so no local id-mapping table exists.
--   * Row Level Security is a BACKSTOP: the backend connects as the Postgres
--     role (table owner, bypasses RLS) and enforces ownership in the API
--     layer. RLS only constrains direct access via the PostgREST anon/
--     authenticated roles.
--   * Deleting an Auth user (admin panel or API) cascades to profiles via the
--     AFTER DELETE trigger, and profiles cascades to conversations/messages
--     via foreign keys.

-- ============================ Tables ============================

create table if not exists public.profiles (
  user_id uuid primary key,
  username text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null unique references public.profiles (user_id) on delete cascade,
  summary text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists ix_conversations_user_id on public.conversations (user_id);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  role varchar(20) not null,
  content text not null,
  sources_json text,
  created_at timestamptz not null default now()
);

create index if not exists ix_messages_conversation_id on public.messages (conversation_id);
create index if not exists ix_messages_conversation_created on public.messages (conversation_id, created_at);

-- ============================ Row Level Security ============================

alter table public.profiles enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = user_id);
create policy "profiles_insert_own" on public.profiles
  for insert with check (auth.uid() = user_id);
create policy "profiles_update_own" on public.profiles
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "profiles_delete_own" on public.profiles
  for delete using (auth.uid() = user_id);

create policy "conversations_select_own" on public.conversations
  for select using (auth.uid() = user_id);
create policy "conversations_insert_own" on public.conversations
  for insert with check (auth.uid() = user_id);
create policy "conversations_update_own" on public.conversations
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "conversations_delete_own" on public.conversations
  for delete using (auth.uid() = user_id);

create policy "messages_select_own" on public.messages
  for select using (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );
create policy "messages_insert_own" on public.messages
  for insert with check (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );
create policy "messages_update_own" on public.messages
  for update using (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );
create policy "messages_delete_own" on public.messages
  for delete using (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );

-- ============================ Auth-user deletion cascade ============================

-- Safety net: when an Auth user is deleted outside the app (e.g. from the
-- dashboard), remove the profile row; the conversations/messages cascade.
create or replace function public.handle_user_deleted()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  delete from public.profiles where user_id = old.id;
  return old;
end;
$$;

drop trigger if exists on_auth_user_deleted on auth.users;
create trigger on_auth_user_deleted
  after delete on auth.users
  for each row execute function public.handle_user_deleted();
