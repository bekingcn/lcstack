from langchain_core.messages import HumanMessage, AIMessage

from lcstack.core.parsers import parse_data_with_type, OutputParserType


def test_pass_through():
    data = "foo"
    assert (
        parse_data_with_type(data=data, output_type=OutputParserType.pass_through)
        == data
    )


def test_message():
    data = "foo"
    message = HumanMessage(content=data)
    assert (
        parse_data_with_type(data=message, output_type=OutputParserType.primitive)
        == data
    )


def test_messages():
    data = "foo"
    messages = [AIMessage(content="something"), HumanMessage(content=data)]
    assert (
        parse_data_with_type(data=messages, output_type=OutputParserType.primitive)
        == data
    )
    # Always last message
    assert (
        parse_data_with_type(
            data=messages, output_type=OutputParserType.primitive, message_index=0
        )
        == data
    )


def test_struct():
    data = "foo"
    _dict = {"foo": data}
    assert (
        parse_data_with_type(
            data=_dict, output_type=OutputParserType.primitive, message_key="foo"
        )
        == data
    )


def test_struct_non_strict():
    data = "foo"
    _dict = {"foo": data}
    assert (
        parse_data_with_type(
            data=_dict,
            strict=False,
            output_type=OutputParserType.primitive,
            message_key="bar",
        )
        == data
    )


def test_struct_strict():
    data = "foo"
    _dict = {"foo": data}
    # will fail with a error
    try:
        parse_data_with_type(
            data=_dict,
            strict=True,
            output_type=OutputParserType.primitive,
            message_key="bar",
        )
    except Exception as e:
        assert isinstance(e, ValueError)
