import pytest
from lcstack.core.parsers.mako import is_mako_expression, get_mako_expression, bind_data


def test_is_mako_expression():
    assert get_mako_expression("%% 'hello'") == "'hello'"
   
def test_not_mako_expression():
    assert get_mako_expression("hello") == None
    
def test_bind_data_plain():
    text = "%% data"
    assert get_mako_expression(text) == "data"
    
    assert bind_data("hello", text=text) == "hello"
    
def test_bind_data_dict():
    assert bind_data({"data": "hello"}, "%% data.data") == "hello"

def test_bind_data_empty_data():
    assert bind_data({}, "%% 1+2") == 3


def test_bind_data_expr_1():
    text = """
    %% 1+2
    """
    parsed = get_mako_expression(text)
    assert parsed == "1+2"
    assert bind_data({}, parsed, parsed=True) == 3

def test_bind_data_expr_2():
    text = """
    %% {
        "data": 1+2,
        "data2": 3+4
    }
    """
    parsed = get_mako_expression(text)
    assert bind_data({}, parsed, parsed=True) == {"data": 3, "data2": 7}
    

def test_bind_data_expr_3():
    text = """
    %% {
        "data": data.data + 2,
        "data2": data.data2 + 4
    }
    """
    parsed = get_mako_expression(text)
    assert bind_data({"data": 1, "data2": 3}, parsed, parsed=True) == {"data": 3, "data2": 7}
    
def test_bind_data_typed_1():
    from langchain_core.messages import AIMessage
    text = """
    %% {
        "content": data.message.content + " value",
        "data2": data.data2 + 4
    }
    """
    parsed = get_mako_expression(text)
    data = {"message": AIMessage(content="foo"), "data2": 3}
    assert bind_data(data, parsed, parsed=True) == {"content": "foo value", "data2": 7}
    

# TODO: more tests

import os
# security issues
# https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Server%20Side%20Template%20Injection/README.md#direct-access-to-os-from-templatenamespace

@pytest.mark.skip(reason="`__import__` is disabled manually")
def test_security_1():
    text = """
    %% __import__("os").getenv('SECRET', "hello")
    """
    parsed = get_mako_expression(text)
    assert bind_data({}, parsed, parsed=True) == os.getenv("SECRET", "hello")
    
    text = """
    %% __import__("os").getenv('APPDATA', "hello")
    """
    parsed = get_mako_expression(text)
    assert bind_data({}, parsed, parsed=True) == os.getenv("APPDATA", "hello22222")

def test_security_2():
    # os.system: cmd to be executed
    text = """
    %% self.module.cache.util.os.system("id")
    """
    parsed = get_mako_expression(text)
    ret = bind_data({}, parsed, parsed=True)
    print("self.module.cache.util.os.system('id'): ", ret)
    assert isinstance(ret, int)