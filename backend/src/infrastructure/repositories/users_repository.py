from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.src.domain.models import User
from backend.src.infrastructure.data.db_client import get_db_session
from backend.src.infrastructure.data.models import User as UserModel


def _to_domain_user(model: UserModel) -> User:
    return User(
        id=model.id,
        username=model.username,
        authentik_sub=model.authentik_sub,
        created_at=model.created_at,
    )


class UsersRepository:
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        with get_db_session() as db:
            model = db.get(UserModel, user_id)
            if model is None:
                return None
            return _to_domain_user(model)

    def get_user_by_sub(self, sub: str) -> Optional[User]:
        with get_db_session() as db:
            stmt = select(UserModel).where(UserModel.authentik_sub == sub)
            model = db.execute(stmt).scalar_one_or_none()
            if model is None:
                return None
            return _to_domain_user(model)

    def get_or_create_user_by_sub(self, sub: str, username: str) -> tuple[int, bool]:
        """Return (user_id, created_new) for an Authentik identity.

        Looks the identity up by its Authentik subject (``sub``). On first
        login a local user row is created; a legacy row that has the same
        username but no subject yet (pre-migration bootstrap users) is adopted
        in place so conversation history is preserved.
        """
        with get_db_session() as db:
            existing_stmt = select(UserModel.id).where(UserModel.authentik_sub == sub)
            existing_id = db.execute(existing_stmt).scalar_one_or_none()
            if existing_id is not None:
                return (existing_id, False)

            legacy_stmt = select(UserModel).where(
                UserModel.username == username, UserModel.authentik_sub.is_(None)
            )
            legacy = db.execute(legacy_stmt).scalar_one_or_none()
            if legacy is not None:
                legacy.authentik_sub = sub
                return (legacy.id, True)

            user = UserModel(username=username, authentik_sub=sub)
            db.add(user)
            try:
                db.flush()
                return (user.id, True)
            except IntegrityError:
                db.rollback()
                return (db.execute(existing_stmt).scalar_one(), False)

    def delete_user_by_id(self, user_id: int) -> bool:
        with get_db_session() as db:
            user = db.get(UserModel, user_id)
            if user is None:
                return False

            db.delete(user)
            return True
