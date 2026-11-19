"""Configuration I/O, including duplicate-key errors before any training starts."""

from pathlib import Path
import yaml

from .registry import ConfigurationError


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # YAML's explicit local override of an anchor is valid.
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ConfigurationError(
                    f"Duplicate YAML key {key!r}, line {key_node.start_mark.line + 1}"
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path):
    path = Path(path).expanduser().resolve()
    try:
        config = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration root must be a mapping")
    return config
