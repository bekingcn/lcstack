from lcstack.core.initializer import RootInitializer
from langchain_core.runnables import Runnable


def make_node_with_mapping(input_mapping: dict = {}, output_mapping: dict = {}):
    config = {
        "pt_node": {
            "initializer": "data_parser",
            "data": {
                "input_type": "pass_through",
                "output_type": "pass_through",
                "input_mapping": input_mapping,
                "output_mapping": output_mapping,
            },
        }
    }
    root = RootInitializer.from_config(config)
    return root.get_ref_initializer("pt_node").build("pt_node").build_original()


def test_is_runnable():
    runnable = make_node_with_mapping()
    assert isinstance(runnable, Runnable)


def test_pass_through():
    runnable = make_node_with_mapping()
    result = runnable.invoke(input="foo")
    print("==>", result)
    assert result == "foo"


def test_im_str2struct():
    im = {"foo": None}
    runnable = make_node_with_mapping(input_mapping=im)

    result = runnable.invoke(input="bar")
    assert result == {"foo": "bar"}


def test_im_struct2str():
    im = {None: "foo"}
    runnable = make_node_with_mapping(input_mapping=im)

    result = runnable.invoke(input={"foo": "bar"})
    assert result == "bar"


def test_om_non_struct_pass_through():
    # TODO: improve this config, add MappingParserArgs for input and output mapping
    # like: output_mapping={"output_type": "primitive"}
    runnable = make_node_with_mapping(
        output_mapping={None: {"name": None, "output_type": "pass_through"}}
    )
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input="bar")
    assert result == "bar"
    result = runnable.invoke(input={"bar": "bar"})
    assert result == {"bar": "bar"}
    result = runnable.invoke(input=AIMessage("bar"))
    assert result == AIMessage("bar")


def test_om_non_struct_to_primitive():
    # TODO: improve this config, add MappingParserArgs for input and output mapping
    # like: output_mapping={"output_type": "primitive"}
    runnable = make_node_with_mapping(
        output_mapping={None: {"name": None, "output_type": "primitive"}}
    )
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input=AIMessage(content="bar"))
    assert result == "bar"
    # list
    result = runnable.invoke(input=["bar", "bar"])
    assert result == "bar\nbar"
    # int
    result = runnable.invoke(input=1)
    assert result == 1
    # list of int, TODO: not supported
    # result = runnable.invoke(input=[1, 2])
    # assert result == "1\n2"


def test_om_non_struct_to_primitive_alt():
    # supporte MappingParserArgs for input and output mapping
    output_mapping = {"output_type": "primitive"}
    runnable = make_node_with_mapping(output_mapping=output_mapping)
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input=AIMessage(content="bar"))
    assert result == "bar"


def test_om_non_struct_to_struct_1():
    runnable = make_node_with_mapping(output_mapping={"result": None})
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input=AIMessage(content="bar"))
    print("==>", result)
    assert result == {"result": AIMessage(content="bar")}


def test_om_non_struct_to_struct_2():
    runnable = make_node_with_mapping(
        output_mapping={"result": {"name": None, "output_type": "primitive"}}
    )
    from langchain_core.messages import AIMessage

    result = runnable.invoke(input=AIMessage(content="bar"))
    print("==>", result)
    assert result == {"result": "bar"}


def test_im_om_simple():
    runnable = make_node_with_mapping(
        input_mapping={"foo": None}, output_mapping={"foo2": "foo"}
    )

    result = runnable.invoke(input="bar")
    assert result == {"foo2": "bar"}
