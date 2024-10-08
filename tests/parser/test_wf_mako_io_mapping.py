from lcstack.core.initializer import RootInitializer


def make_node_with_mapping(
    wf_input_expr: str = None,
    schema: list = [],
    input_expr: str = None,
    output_expr: str = None,
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
                    "input_expr": wf_input_expr,
                    "vertices": [
                        {
                            "name": "pt_node",
                            "agent": "{{ pt_node }}",
                            "input_expr": input_expr,
                            "output_expr": output_expr,
                        }
                    ],
                }
            },
        },
    }
    root = RootInitializer.from_config(config)
    return root.get_ref_initializer("wf_node").build("wf_node").build_original()


config = {"thread_id": "wf_node_mapping_123"}


def test_wf_im_any2struct_str():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        wf_input_expr="%% {'foo': data}",
    )
    result = runnable.invoke(input="foo value", config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_wf_im_any2struct_msg_content():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        wf_input_expr="%% {'foo': data.content}",
    )
    from langchain_core.messages import HumanMessage

    result = runnable.invoke(input=HumanMessage("foo value"), config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value"}


def test_node_im_om_struct2str2struct():
    runnable = make_node_with_mapping(
        schema=[{"name": "foo", "field_type": "str"}],
        input_expr="%% data.foo + ' value'",
        output_expr="%% {'foo': data}",
    )
    result = runnable.invoke(input={"foo": "foo value"}, config=config)
    result.pop("branch__steps__")
    assert result == {"foo": "foo value value"}


# TODO: more test cases
