import os


def require(key: str) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable '{key}' is not set")
    return value


def require_int(key: str) -> int:
    value = require(key)
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"Environment variable '{key}' must be an integer, got: {value}")


def require_float(key: str) -> float:
    value = require(key)
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"Environment variable '{key}' must be a float, got: {value}")


def get_int(key: str, default: int) -> int:
    """Get an integer environment variable with a default value."""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_str(key: str, default: str) -> str:
    """Get a string environment variable with a default value."""
    return os.getenv(key, default)