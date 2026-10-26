"""Test-only constructor seam for an omitted episodic-store path."""
from functools import wraps


def private_default_store_init(original, private_path):
    @wraps(original)
    def isolated_init(self, *args, **kwargs):
        if not args and "db_path" not in kwargs:
            kwargs["db_path"] = private_path
        return original(self, *args, **kwargs)

    return isolated_init
