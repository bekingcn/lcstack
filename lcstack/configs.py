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