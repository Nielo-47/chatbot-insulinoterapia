from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from backend.src.db.models import User
from backend.src.db.session import get_db_session


class UsersRepository:
    def get_user_id(self, username: str) -> Optional[int]:
        with get_db_session() as db:
            stmt = select(User.id).where(User.username == username)
            return db.execute(stmt).scalar_one_or_none()

    def get_or_create_user_id(self, username: str, hashed_password: str) -> int:
        with get_db_session() as db:
            existing_stmt = select(User.id).where(User.username == username)
            existing_id = db.execute(existing_stmt).scalar_one_or_none()
            if existing_id is not None:
                return existing_id

            user = User(username=username, hashed_password=hashed_password)
            db.add(user)
            try:
                db.flush()
                return user.id
            except IntegrityError:
                db.rollback()
                return db.execute(existing_stmt).scalar_one()

    def delete_user_by_id(self, user_id: int) -> bool:
        with get_db_session() as db:
            user = db.get(User, user_id)
            if user is None:
                return False

            db.delete(user)
            return True
