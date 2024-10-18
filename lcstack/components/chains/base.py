
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from langchain.chains.llm import LLMChain
from langchain_community.chains.llm_requests import LLMRequestsChain
from ..utils import keyed_value_runnable
from ..output_parser import create_output_parser, SUPPORTED_OUTPUT_PARSERS


def load_llm_requests_chain(llm, prompt):
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = LLMRequestsChain(llm_chain=llm_chain)
    return chain


def create_llm_chain(llm: Runnable, prompt: Runnable, output_type: str = "str", output_key: str = "output") -> Runnable:
    """Create a LLM chain.

    Args:
        llm: LLM to use
        prompt: Prompt to use
        output_type: Output type to use
        output_key: Output key to use

    Returns:
        LCEL chain

    """

    # following the llm node patterns
    if output_type in SUPPORTED_OUTPUT_PARSERS:
        post_parser = create_output_parser(type=output_type, output_key=output_key)
    elif output_type == "message":
        post_parser = keyed_value_runnable(key=output_key)
    elif output_type == "messages":
        # TODO: improve this
        post_parser = (lambda x: [x]) | keyed_value_runnable(key=output_key)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    
    return (prompt | llm | post_parser).with_config(name="llm_chain")
