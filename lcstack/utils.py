# draw the graph with mermaid
import base64
from enum import Enum
import os
import re
from typing import Callable, Optional
import yaml

## Parsing the YAML config
    
def load_flatten_settings(settings_file: str, dict_as_value: bool = False, tags: dict = {}):
    class _Loader(yaml.SafeLoader):
        pass

    for tag in tags:
        _Loader.add_implicit_resolver(tag, pattern, None)
        _Loader.add_constructor(tag, constructor_variables_func(tags[tag]))
    settings = yaml.load(open(settings_file, "r"), Loader=_Loader)
    # flatten settings, and convert to dict, connect keys with "." as separator for nested keys
    def _flatten(d, parent_key="", sep="."):
        d = d or {}
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                # TODO: Should support dict as a object too?
                if dict_as_value:
                    items.append((new_key, v))
                items.extend(_flatten(v, new_key, sep=sep).items())
            # TODO: handle lists, list will be as plain object
            else:
                items.append((new_key, v))
        return dict(items)
    settings = _flatten(settings)
    return settings

# pattern for global vars: look for ${word}
pattern = re.compile('.*?\${(\s*[\w.]+\s*)}.*?')

# the tag will be used to mark where to start searching for the pattern
# e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
def constructor_variables_func(vars):
    # support both dict and callable
    if isinstance(vars, (dict, os._Environ)):
        _get_item = vars.get
    elif isinstance(vars, Callable):
        _get_item = vars
    else:
        raise Exception(f'Not supported type to get item from: {type(vars)}')
    def _func(loader, node):            
        node: yaml.ScalarNode = node
        loader : yaml.SafeLoader = loader
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in this line
        if match:
            full_value = value
            for g in match:
                replaced_value = _get_item(g.strip(), g)
                if isinstance(replaced_value, str):
                    full_value = full_value.replace(
                        f'${{{g}}}', _get_item(g.strip(), g)
                    )
                else:
                    # replaced_value could be a list, dict, or primitive value
                    if f'${{{g}}}' == value:
                        full_value = replaced_value
                    else:
                        raise Exception(f'Cannot replace value with dict or list: {value}')
            return full_value
        return value
    return _func

# !ENV, !CONF, !REF ...
def parse_config(data=None, tags={}):
    """
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:

    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'

    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    
    class _Loader(yaml.SafeLoader):
        pass

    for tag in tags:
        _Loader.add_implicit_resolver(tag, pattern, None)
        _Loader.add_constructor(tag, constructor_variables_func(tags[tag]))
    return yaml.load(data, Loader=_Loader)

## Drawing graphs with Mermaid

def _escape_node_label(node_label: str) -> str:
    """Escapes the node label for Mermaid syntax."""
    return re.sub(r"[^0-9a-zA-Z-_:#]", "_", node_label)

# function to draw the mermaid graph
def draw_mermaid(nodes, edges):
    # create the mermaid
    mermaid_graph = "graph TD;\n"
    for n, v in nodes.items():
        mermaid_graph += f"{_escape_node_label(n)}[\"`{v}`\"];\n"
    edge_label = " --> "
    for e in edges:
        source, target, label = e
        # label = "default"
        mermaid_graph += (
            f"\t{_escape_node_label(source)}{edge_label}"
            f"|{label}|"
            f"{_escape_node_label(target)};\n"
        )
    return mermaid_graph

class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid"""

    PYPPETEER = "pyppeteer"  # Uses Pyppeteer to render the graph
    API = "api"  # Uses Mermaid.INK API to render the graph

def _render_mermaid_using_api(
    mermaid_syntax: str,
    output_file_path: Optional[str] = None,
    background_color: Optional[str] = "white",
) -> bytes:
    """Renders Mermaid graph using the Mermaid.INK API."""
    try:
        import requests  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        ) from e

    # Use Mermaid API to render the image
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode(
        "ascii"
    )

    # Check if the background color is a hexadecimal color code using regex
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

    image_url = (
        f"https://mermaid.ink/img/{mermaid_syntax_encoded}?bgColor={background_color}"
    )
    response = requests.get(image_url)
    if response.status_code == 200:
        img_bytes = response.content
        if output_file_path is not None:
            with open(output_file_path, "wb") as file:
                file.write(response.content)

        return img_bytes
    else:
        raise ValueError(
            f"Failed to render the graph using the Mermaid.INK API. "
            f"Status code: {response.status_code}."
        )

def draw_mermaid_png(
    mermaid_syntax: str,
    output_file_path: Optional[str] = None,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,  # pyppeteer
    background_color: Optional[str] = "white",
    padding: int = 10,
) -> bytes:
    """Draws a Mermaid graph as PNG using provided syntax."""
    if draw_method == MermaidDrawMethod.PYPPETEER:
        import asyncio

        img_bytes = asyncio.run(
            _render_mermaid_using_pyppeteer(
                mermaid_syntax, output_file_path, background_color, padding
            )
        )
    elif draw_method == MermaidDrawMethod.API:
        img_bytes = _render_mermaid_using_api(
            mermaid_syntax, output_file_path, background_color
        )
    else:
        supported_methods = ", ".join([m.value for m in MermaidDrawMethod])
        raise ValueError(
            f"Invalid draw method: {draw_method}. "
            f"Supported draw methods are: {supported_methods}"
        )

    return img_bytes