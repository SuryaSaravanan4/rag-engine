import os


def require_env(var_name: str) -> str:
    """Return the value of an environment variable, or raise a clear error if unset."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"{var_name} is not set. Set it in your environment or .env.local, "
            f"or pass the api_key argument explicitly."
        )
    return value
