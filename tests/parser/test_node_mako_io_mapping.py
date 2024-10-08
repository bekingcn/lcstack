from lcstack.core.initializer import RootInitializer


def make_node_with_mapping(input_expr: str = "", output_expr: str = ""):
    from lcstack.configs import set_expr_enabled

    set_expr_enabled(True)
    config = {
        "pt_node": {
            "initializer": "data_parser",
            "data": {
                "input_type": "pass_through",
                "output_type": "pass_through",
                "input_expr": input_expr,
                "output_expr": output_expr,
            },
        }
    }
    root = RootInitializer.from_config(config)
    return root.get_ref_initializer("pt_node").build("pt_node").build_original()


def test_pass_through():
    runnable = make_node_with_mapping()
    result = runnable.invoke(input="foo")
    print("==>", result)
    assert result == "foo"


def test_im_str2struct():
    runnable = make_node_with_mapping(input_expr="%% {'foo': data}")

    result = runnable.invoke(input="bar")
    assert result == {"foo": "bar"}


def test_im_struct2str():
    runnable = make_node_with_mapping(input_expr="%% data.foo")

    result = runnable.invoke(input={"foo": "bar"})
    assert result == "bar"


def test_om_non_struct_pass_through():
    runnable = make_node_with_mapping(output_expr="%% data")
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input="bar")
    assert result == "bar"
    result = runnable.invoke(input=AIMessage("bar"))
    assert result == AIMessage("bar")
    result = runnable.invoke(input=1)
    assert result == 1
    # NOTE: not direct value for dict, failed to pass through
    # should be:
    runnable = make_node_with_mapping(output_expr="%%{'bar': data.bar}")
    result = runnable.invoke(input={"bar": "bar"})
    assert result == {"bar": "bar"}


def test_om_non_struct_to_primitive():
    runnable = make_node_with_mapping(output_expr="%% data.content")
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input=AIMessage(content="bar"))
    assert result == "bar"

    # list
    # should be `\\`
    runnable = make_node_with_mapping(output_expr="%% '\\n'.join(data)")
    result = runnable.invoke(input=["bar", "bar"])
    assert result == "bar\nbar"

    # int
    runnable = make_node_with_mapping(
        output_expr="%% '\\n'.join([str(x) for x in data])"
    )
    result = runnable.invoke(input=[1, 2])
    assert result == "1\n2"


def test_im_om_simple():
    runnable = make_node_with_mapping(
        input_expr="%% {'foo': data}", output_expr="%% {'foo2': data.foo}"
    )

    result = runnable.invoke(input="bar")
    assert result == {"foo2": "bar"}
