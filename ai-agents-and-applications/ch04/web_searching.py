from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List


def web_search(query: str, num_results: int) -> List[str]:
    return [r["link"] for r in DuckDuckGoSearchAPIWrapper().results(query, num_results)]
