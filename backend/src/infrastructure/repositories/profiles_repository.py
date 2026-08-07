import logging
import uuid
from typing import Optional, Tuple

from sqlalchemy.exc import IntegrityError

from backend.src.domain.models import Profile
from backend.src.infrastructure.data.db_client import get_db_session
from backend.src.infrastructure.data.models import Profile as ProfileModel

logger = logging.getLogger(__name__)


def _to_domain_profile(model: ProfileModel) -> Profile:
    return Profile(
        user_id=model.user_id,
        username=model.username,
        created_at=model.created_at,
    )


class ProfilesRepository:
    def get_profile_by_id(self, user_id: uuid.UUID) -> Optional[Profile]:
        with get_db_session() as db:
            model = db.get(ProfileModel, user_id)
            if model is None:
                return None
            return _to_domain_profile(model)

    def get_or_create_profile(self, user_id: uuid.UUID, username: str) -> Tuple[uuid.UUID, bool]:
        """Return (user_id, created_new) for a Supabase identity.

        The profile primary key IS the Supabase Auth user id (the JWT ``sub``
        claim), so this is an upsert rather than a lookup-by-sub mapping. The
        username follows the access token's email claim.
        """
        with get_db_session() as db:
            if db.get(ProfileModel, user_id) is not None:
                return (user_id, False)

            profile = ProfileModel(user_id=user_id, username=username)
            db.add(profile)
            try:
                db.flush()
                return (user_id, True)
            except IntegrityError:
                db.rollback()
                return (user_id, False)

    def delete_profile(self, user_id: uuid.UUID) -> bool:
        """Delete a profile row (conversations/messages cascade)."""
        with get_db_session() as db:
            profile = db.get(ProfileModel, user_id)
            if profile is None:
                return False
            db.delete(profile)
            return True
