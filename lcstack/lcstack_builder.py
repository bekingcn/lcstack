import yaml, re, os
from pathlib import Path

from .configs import get_config_root
from .utils import parse_config, load_flatten_settings

DEFAULT_SETTINGS_FILE = "settings.yaml"
DEFAULT_TEMPLATE_FILE = "templates.yaml"
DEFAULT_SECRET_FILE = ".secret.yaml"
FIELD_NAME_INCLUDE = "include"
TAG_NAME_ENV = "!ENV"
TAG_NAME_SECRET = "!SECRET"
TAG_NAME_SET = "!SET"
TAG_NAME_INCLUDE = "!INC"


class YamlBuilder:
    def __init__(self) -> None:
        self.template = None
        self.config_file = None
        self.config = None
        self.env = False
        self.secret = False
        self.settings_file = None
        self.other_config = None

    @classmethod
    def from_yaml(cls, yaml: str):
        """Construct from a yaml file."""
        instance = cls()
        instance.config_file = get_config_root() / yaml
        return instance

    @classmethod
    def from_object(cls, obj):
        """Construct from an str or dict object."""
        instance = cls()
        instance.config = obj
        return instance

    def with_template(self, template: str = get_config_root() / DEFAULT_TEMPLATE_FILE):
        self.template = template
        return self

    def with_env(self, env: bool = True):
        self.env = env
        return self

    def with_secret(self, secret: bool = True):
        self.secret = secret
        return self

    def with_settings(self, settings: str | bool = True):
        settings_file = settings
        if isinstance(settings_file, str):
            self.settings_file = get_config_root() / settings_file
        elif settings_file is True:
            settings_file = get_config_root() / DEFAULT_SETTINGS_FILE
            if settings_file.exists():
                self.settings_file = settings_file
            elif (Path.cwd() / DEFAULT_SETTINGS_FILE).exists():
                self.settings_file = Path.cwd() / DEFAULT_SETTINGS_FILE
            else:
                raise ValueError("settings file not found")
        return self

    def merge_with(self, other: dict | None):
        # check other is dict
        if other and not isinstance(other, dict):
            raise ValueError("other config to merge must be a dict")
        self.other_config = other
        return self

    def _merge_config(self, config):
        from copy import deepcopy

        _other_config = deepcopy(self.other_config)

        # here we merge extra_config with config recursively
        def merge_dicts(source, target):
            for key, val in source.items():
                if (
                    isinstance(val, dict)
                    and key in target
                    and isinstance(target[key], dict)
                ):
                    merge_dicts(source[key], target[key])
                else:
                    target[key] = source[key]

        merge_dicts(config, _other_config)
        return config

    def _build_config(self):
        tags = {}

        # support env and secret
        if self.env:
            from dotenv import load_dotenv

            load_dotenv()
            tags[TAG_NAME_ENV] = os.environ
        if self.secret:
            secret_path = Path.cwd() / DEFAULT_SECRET_FILE
            if secret_path.exists():
                tags[TAG_NAME_SECRET] = load_flatten_settings(
                    Path.cwd() / DEFAULT_SECRET_FILE
                )
            else:
                print(
                    f"Warning: secret file {secret_path} file not found, it will be ignored"
                )
        # support settings file
        if self.settings_file or self.template:
            # NOTE: here we load settings with env and secrets, and also expose all dicts and lists in settings
            tags[TAG_NAME_SET] = (
                load_flatten_settings(self.settings_file, dict_as_value=True, tags=tags)
                if self.settings_file
                else {}
            )
            if self.template:
                templates = load_flatten_settings(
                    self.template, dict_as_value=True, tags=tags
                )
                # merge templates with settings which may contain the templates key
                tags[TAG_NAME_SET]["templates"] = {
                    **tags[TAG_NAME_SET].get("templates", {}),
                    **templates,
                }

        def _get_include(relative_path, default=None):
            if os.path.isfile(config_dir / relative_path):
                with open(config_dir / relative_path, "r") as f:
                    config = parse_config(f, tags=tags)
                return config
            else:
                raise FileExistsError(
                    f"{relative_path} not found or is not a file in {config_dir}"
                )

        tags[TAG_NAME_INCLUDE] = _get_include

        # load config from file
        if self.config_file:
            # always support including other yaml files,
            # for now only support configs in the same directory or sub-directories
            config_dir = Path(self.config_file).parent
            with open(self.config_file, "r") as f:
                config = parse_config(f, tags=tags)
        # config is string, parse it with tags
        elif self.config and isinstance(self.config, str):
            # include other config files under current directory
            config_dir = Path.cwd()
            config = parse_config(data=self.config, tags=tags)
        # config is a dict and tags are provided, re-generate config with tags
        elif self.config and tags and isinstance(self.config, dict):
            # in this case, support `include` only, tags is not supported
            config_dir = Path.cwd()
            # no need to re-generate config
            config = self.config  # parse_config(data=yaml.dump(self.config), tags=tags)
        else:
            raise ValueError("config or config_file must be provided")

        # support `include` to include and merge other config files
        if FIELD_NAME_INCLUDE in config:
            include_files = config.pop(FIELD_NAME_INCLUDE)
            if isinstance(include_files, str):
                include_files = [include_files]
            if not isinstance(include_files, list):
                raise ValueError("include must be a string or a list of strings")
            merged_config = {}
            for include in include_files:
                with open(config_dir / include, "r") as f:
                    include_config = parse_config(f, tags=tags)
                # NOTE: to avoid confusion, support shallow override for included config
                merged_config.update(include_config)
            # currrent config will override the merged config
            merged_config.update(config)
            config = merged_config

        if self.other_config:
            config = self._merge_config(config)

        return config

    # will be implemented by subclasses
    def build(self) -> any:
        pass

    def pprint(self, print=True):
        if print:
            from pprint import pprint

            config = self._build_config()
            pprint(config)
        return self
