from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.tools import tool

def create_tool_scrape_webpages():
    @tool
    def scrape_webpages(urls: list) -> str:
        """Use requests and bs4 to scrape the provided web pages for detailed information."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join(
            [
                f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
    return scrape_webpages