import os
import pytest
from lcstack.core.parsers.mako import get_mako_expression, bind_data


def test_is_mako_expression():
    assert get_mako_expression("%% 'hello'") == "'hello'"


def test_not_mako_expression():
    assert get_mako_expression("hello") is None


def test_bind_data_plain():
    text = "%% data"
    assert get_mako_expression(text) == "data"

    assert bind_data(text=text, data="hello") == "hello"


def test_bind_data_dict():
    assert (
        bind_data(
            "%% data.data",
            {"data": "hello"},
        )
        == "hello"
    )


def test_bind_data_empty_data():
    assert (
        bind_data(
            "%% 1+2",
            {},
        )
        == 3
    )


def test_bind_data_expr_1():
    text = """
    %% 1+2
    """
    parsed_text = get_mako_expression(text)
    assert parsed_text == "1+2"
    assert bind_data(parsed_text, {}, parsed=True) == 3


def test_bind_data_expr_2():
    text = """
    %% {
        "data": 1+2,
        "data2": 3+4
    }
    """
    parsed_text = get_mako_expression(text)
    assert bind_data(parsed_text, {}, parsed=True) == {"data": 3, "data2": 7}


def test_bind_data_expr_3():
    text = """
    %% {
        "data": data.data + 2,
        "data2": data.data2 + 4
    }
    """
    parsed_text = get_mako_expression(text)
    assert bind_data(parsed_text, {"data": 1, "data2": 3}, parsed=True) == {
        "data": 3,
        "data2": 7,
    }


def test_bind_data_typed_1():
    from langchain_core.messages import AIMessage

    text = """
    %% {
        "content": data.message.content + " value",
        "data2": data.data2 + 4
    }
    """
    parsed_text = get_mako_expression(text)
    data = {"message": AIMessage(content="foo"), "data2": 3}
    assert bind_data(parsed_text, data, parsed=True) == {
        "content": "foo value",
        "data2": 7,
    }


def test_bind_data_typed_function_1():
    from langchain_core.messages import AIMessage

    text = """
    %% {
        "message": AIMessage(content="foo"),
    }
    """
    parsed_text = get_mako_expression(text)
    data = {}
    assert bind_data(parsed_text, data, parsed=True) == {
        "message": AIMessage(content="foo")
    }


# TODO: more tests
# security issues
# https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Server%20Side%20Template%20Injection/README.md#direct-access-to-os-from-templatenamespace


@pytest.mark.skip(reason="`__import__` is disabled manually")
def test_security_1():
    text = """
    %% __import__("os").getenv('SECRET', "hello")
    """
    parsed_text = get_mako_expression(text)
    assert bind_data(parsed_text, {}, parsed=True) == os.getenv("SECRET", "hello")

    text = """
    %% __import__("os").getenv('APPDATA', "hello")
    """
    parsed_text = get_mako_expression(text)
    assert bind_data(parsed_text, {}, parsed=True) == os.getenv("APPDATA", "hello22222")


def test_security_2():
    # os.system: cmd to be executed
    text = """
    %% self.module.cache.util.os.system("id")
    """
    parsed_text = get_mako_expression(text)
    ret = bind_data(parsed_text, {}, parsed=True)
    print("self.module.cache.util.os.system('id'): ", ret)
    assert isinstance(ret, int)
