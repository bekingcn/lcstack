from lcstack import LcStack
from langchain_core.runnables import Runnable


def make_node_with_mapping(
    schema: list = [],
    default_setters: list = [],
    branchs: list = [],
):
    from lcstack.configs import set_expr_enabled

    set_expr_enabled(True)
    config = {
        "pt_node": {
            "initializer": "data_parser",
            "data": {
                "input_type": "pass_through",
                "output_type": "pass_through",
            },
        },
        "wf_node": {
            "initializer": "conditional_graph",
            "data": {
                "inline": {
                    "name": "wf_node_test",
                    "schema": schema,
                    "vertices": [
                        {
                            "name": "pt_node",
                            "agent": "{{ pt_node }}",
                            "next": "branch_0",
                        },
                        {
                            "name": "branch_0",
                            "default": "__end__",
                            "default_setters": default_setters,
                            "branchs": branchs,
                        },
                    ],
                }
            },
        },
    }
    root = LcStack.from_config(config)
    runnable = root.get_ref_initializer("wf_node").build("wf_node").build()
    # save a image
    # open("test_wf_mako_in_branchs.png", "wb").write(runnable.get_graph().draw_mermaid_png())
    return runnable


config = {"thread_id": "wf_node_mako_123"}


def test_is_runnable():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "type": "string"}],
    )
    assert isinstance(runnable, Runnable)


def test_pass_through():
    """NOTE: INVALID case for workflow"""
    runnable = make_node_with_mapping(
        schema=[],
    )
    result = runnable.invoke(input={}, config=config)
    # NOTE: caused by: pass_through, possibly introduces `branch__steps__`
    result.pop("branch__steps__")
    assert result == {}


def test_setter_plain():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "type": "string"}],
        default_setters=[{"property": "foo", "value": "foo value"}],
    )
    result = runnable.invoke(input={}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_setter_mako_1():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "type": "string"}],
        default_setters=[{"property": "foo", "value": "%% data.foo + ' value'"}],
    )
    result = runnable.invoke(input={"foo": "foo"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_setter_mako_2():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "type": "string"}, {"name": "bar", "type": "string"}],
        default_setters=[{"property": "bar", "value": "%% data.foo + ' value'"}],
    )
    result = runnable.invoke(input={"foo": "foo"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo", "bar": "foo value"}


def test_matching_setter_mako_matched():
    runnable = make_node_with_mapping(
        schema=[
            {"name": "foo", "type": "string"},
            {"name": "bar", "type": "string"},
            {"name": "matching", "type": "string"},
        ],
        default_setters=[{"property": "bar", "value": "%% data.foo + ' value'"}],
        branchs=[
            {
                "conditions": "%% data.foo == 'foo'",
                "next": "__end__",
                "setters": [{"property": "matching", "value": "true"}],
            }
        ],
    )
    result = runnable.invoke(input={"foo": "foo"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo", "matching": "true", "bar": ""}


def test_matching_setter_mako_not_matched():
    runnable = make_node_with_mapping(
        schema=[
            {"name": "foo", "type": "string"},
            {"name": "bar", "type": "string"},
            {"name": "matching", "type": "string"},
        ],
        default_setters=[{"property": "bar", "value": "%% data.foo + ' value'"}],
        branchs=[
            {
                "conditions": "%% data.foo == 'bar'",
                "next": "__end__",
                "setters": [{"property": "matching", "value": "true"}],
            }
        ],
    )
    result = runnable.invoke(input={"foo": "foo"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo", "matching": "", "bar": "foo value"}


# TODO: more test cases
