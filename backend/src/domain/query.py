from typing import Literal


QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


__all__ = ["QueryMode"]