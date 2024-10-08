from io import StringIO
from mako.template import Template
from mako.runtime import Context
from typing import Any


def is_mako_expression(text: str) -> bool:
    return get_mako_expression(text) is not None


def get_mako_expression(text: str) -> str:
    parsed = text
    # strip prefix characters like: \n, space, etc
    parsed = parsed.strip("\n\t ")
    # strip %%
    if not parsed.startswith("%%"):
        return None
    parsed = parsed[2:]
    return parsed.strip()


class _MakoWrapper:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def bind_functions(text):
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    return {
        "AIMessage": AIMessage,
        "HumanMessage": HumanMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
    }


# TODO: which types should be supported here? str, dict, message, list(?) etc
def bind_data(text: str, data: Any, variables={}, parsed: bool = False) -> str:
    _text = text
    if isinstance(data, dict):
        bound = _MakoWrapper(**data)
    else:
        bound = data
    # rebuild the text
    if not parsed:
        _text = get_mako_expression(_text)
    # NOTE: directly return the text if it's not a mako expression
    if not _text:
        return text
    # TODO: validate the text for security issues: [ "__", "exec", "str" , "import", ... ]
    # _text = f"<% _.value = {_text} %>"
    mako_template = Template(
        _text,
        # add prefix with preprocessor
        preprocessor=lambda text: f"<% _.value = {text} %>",
    )
    ret_value = _MakoWrapper(value=None)

    outer_data = {
        "data": bound,
        "vars": variables,
        "_": ret_value,
        "__import__": None,
        # "self": _MakoWrapper(value=None),
        # add message functions or something
        **bind_functions(_text),
    }

    print("mako text", text)
    context = Context(StringIO(text), **outer_data)
    # context.write(text)
    mako_template.render_context(context)
    return ret_value.value


def eval_expr(expr: str, data: Any, variables={}) -> Any:
    return bind_data(data=data, text=expr, variables=variables, parsed=False)
