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