from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable
from langchain_core.prompts import BasePromptTemplate
from langchain_core.language_models import BaseLanguageModel

from langchain.chains.sql_database.query import (
    create_sql_query_chain as _create_sql_query_chain,
)
from langchain_community.utilities.sql_database import SQLDatabase

from ..utils import keyed_value_runnable


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db_uri: str,
    db_engine_args: Optional[dict] = None,
    db_kwargs: Optional[dict] = None,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Dict[str, Any], str]:
    """Create a chain that generates SQL queries."""
    _engine_args = db_engine_args or {}
    _db_kwargs = db_kwargs or {}
    db = SQLDatabase.from_uri(db_uri, engine_args=_engine_args, **_db_kwargs)
    return _create_sql_query_chain(llm=llm, db=db, prompt=prompt, k=k) | keyed_value_runnable(key="sql")
