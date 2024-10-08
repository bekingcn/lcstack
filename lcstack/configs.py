import os
from pathlib import Path

_config_root = None


def set_config_root(config_root: str):
    global _config_root
    if not Path(config_root).is_absolute():
        _config_root = Path.cwd() / config_root
    else:
        _config_root = Path(config_root)


def get_config_root():
    if _config_root:
        return _config_root
    # from env
    if "LCSTACK_CONFIG_ROOT" in os.environ:
        return Path(os.environ["LCSTACK_CONFIG_ROOT"])
    # not config root, use cwd
    return Path.cwd()


_expr_enabled = None


def get_expr_enabled():
    if _expr_enabled is not None:
        return _expr_enabled
    if "LCSTACK_EXPR_ENABLED" in os.environ:
        return os.environ.get("LCSTACK_EXPR_ENABLED", "false").lower() == "true"
    return False


def set_expr_enabled(expr_enabled: bool):
    global _expr_enabled
    _expr_enabled = expr_enabled
    if expr_enabled:
        print(
            "WARNING: LCSTACK_EXPR_ENABLED is set to true. Use at your own risk because it is not secure for external use."
        )
