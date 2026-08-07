"""Database schema definitions for profiles, conversations, and messages.

The database is Supabase Postgres. Primary keys are UUIDs, and the profile
primary key doubles as the Supabase Auth user id (``auth.users.id``) exposed in
the ``sub`` claim of the access token, so no local mapping table is needed.

Data is created/dropped via the Supabase SQL editor (see
scripts/supabase_migration.sql); ``Base.metadata`` exists so the integration
tests can build the same schema in an isolated schema, not for runtime DDL.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for all ORM models."""


class Profile(Base):
    __tablename__ = "profiles"

    # Equal to Supabase Auth user id (the JWT ``sub`` claim). No FK to
    # auth.users here: the auth schema is owned by Supabase and the reference
    # is enforced by convention plus an AFTER DELETE trigger in the migration
    # script (see scripts/supabase_migration.sql).
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=func.gen_random_uuid())
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("profiles.user_id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=func.gen_random_uuid())
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


Index("ix_messages_conversation_created", Message.conversation_id, Message.created_at)
