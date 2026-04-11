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
        hashed_password=model.hashed_password,
        created_at=model.created_at,
    )


class UsersRepository:
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        with get_db_session() as db:
            model = db.get(UserModel, user_id)
            if model is None:
                return None
            return _to_domain_user(model)

    def get_user_by_username(self, username: str) -> Optional[User]:
        with get_db_session() as db:
            stmt = select(UserModel).where(UserModel.username == username)
            model = db.execute(stmt).scalar_one_or_none()
            if model is None:
                return None
            return _to_domain_user(model)

    def get_or_create_user_id(self, username: str, hashed_password: str) -> int:
        with get_db_session() as db:
            existing_stmt = select(UserModel.id).where(UserModel.username == username)
            existing_id = db.execute(existing_stmt).scalar_one_or_none()
            if existing_id is not None:
                return existing_id

            user = UserModel(username=username, hashed_password=hashed_password)
            db.add(user)
            try:
                db.flush()
                return user.id
            except IntegrityError:
                db.rollback()
                return db.execute(existing_stmt).scalar_one()

    def delete_user_by_id(self, user_id: int) -> bool:
        with get_db_session() as db:
            user = db.get(UserModel, user_id)
            if user is None:
                return False

            db.delete(user)
            return True
