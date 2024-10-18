from lcstack.core.initializer import RootInitializer
from langchain_core.runnables import Runnable


def make_node_with_mapping(
    wf_input_mapping: dict = {},
    schema: list = [],
    input_mapping: dict = {},
    output_mapping: dict = {},
):
    config = {
        "pt_node": {
            "initializer": "data_parser",
            "data": {
                "input_type": "pass_through",
                "output_type": "pass_through",
                # "input_mapping": input_mapping,
                # "output_mapping": output_mapping
            },
        },
        "wf_node": {
            "initializer": "conditional_graph",
            "data": {
                "inline": {
                    "name": "wf_node_test",
                    "schema": schema,
                    "input_mapping": wf_input_mapping,
                    "vertices": [
                        {
                            "name": "pt_node",
                            "agent": "{{ pt_node }}",
                            "input_mapping": input_mapping,
                            "output_mapping": output_mapping,
                        }
                    ],
                }
            },
        },
    }
    root = RootInitializer.from_config(config)
    return root.get_ref_initializer("wf_node").build("wf_node").build()


config = {"thread_id": "wf_node_mapping_123"}


def test_is_runnable():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "type": "string"}],
        wf_input_mapping={"foo": None},
    )
    assert isinstance(runnable, Runnable)


def test_pass_through():
    """NOTE: INVALID case for workflow"""
    runnable = make_node_with_mapping(
        schema=[],
        wf_input_mapping={},
    )
    result = runnable.invoke(input={}, config=config)
    # NOTE: caused by: pass_through, possibly introduces `branch__steps__`
    result.pop("branch__steps__")
    assert result == {}


def test_pass_through_struct():
    """NOTE: INVALID case for workflow"""
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        wf_input_mapping={},
    )
    result = runnable.invoke(input={"foo": "foo value"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_wf_im_any2struct_str():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        wf_input_mapping={"foo": None},
    )
    result = runnable.invoke(input="foo value", config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}

    from langchain_core.messages import HumanMessage

    result = runnable.invoke(input=HumanMessage("foo value"), config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_wf_im_any2struct_message_1():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "message"}],
        wf_input_mapping={"foo": None},
    )
    from langchain_core.messages import HumanMessage

    result = runnable.invoke(input="foo value", config=config)
    result.pop("branch__steps__")
    assert result == {"foo": HumanMessage("foo value", name="input")}


def test_wf_im_any2struct_message_2():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "message"}],
        wf_input_mapping={
            "foo": {"name": None, "output_type": "message", "message_role": "ai"},
        },
    )
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input="foo value", config=config)
    result.pop("branch__steps__")
    assert result == {"foo": AIMessage("foo value")}


def test_wf_im_any2struct_messages():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "messages"}],
        wf_input_mapping={"foo": None},
    )
    from langchain_core.messages import HumanMessage

    result = runnable.invoke(input="foo value", config=config)
    result.pop("branch__steps__")
    assert result == {"foo": [HumanMessage("foo value", name="input")]}


def test_node_im_struct2str():
    im = {None: "foo"}
    om = {"foo": None}
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        input_mapping=im,
        output_mapping=om,
    )

    result = runnable.invoke(input={"foo": "foo value"}, config=config)
    result.pop("branch__steps__", None)
    assert result == {"foo": "foo value"}


# TODO: more test cases
