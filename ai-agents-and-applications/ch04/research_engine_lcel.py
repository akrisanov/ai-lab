from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from llm_models import get_llm
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
    RESEARCH_REPORT_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    WEB_SEARCH_PROMPT_TEMPLATE,
)
from utils import to_obj
from web_searching import web_search
from web_scraping import web_scrape

question = "What can I see and do in the Spanish town of Astorga?"

# Assistant Instructions chain

assistant_instructions_chain = (
    {"user_question": RunnablePassthrough()}
    | ASSISTANT_SELECTION_PROMPT_TEMPLATE
    | get_llm()
    | StrOutputParser()
    | to_obj
)

# Web Searches chain

NUM_SEARCH_QUERIES = 2

web_searches_chain = (
    RunnableLambda(
        lambda x: {
            "assistant_instructions": x["assistant_instructions"],
            "num_search_queries": NUM_SEARCH_QUERIES,
            "user_question": x["user_question"],
        }
    )
    | WEB_SEARCH_PROMPT_TEMPLATE
    | get_llm()
    | StrOutputParser()
    | to_obj
)

# Search and Summarization chain

## Search result urls

NUM_SEARCH_RESULTS_PER_QUERY = 3

search_result_urls_chain = RunnableLambda(
    lambda x: [
        {
            "result_url": url,
            "search_query": x["search_query"],
            "user_question": x["user_question"],
        }
        for url in web_search(
            query=x["search_query"],
            num_results=NUM_SEARCH_RESULTS_PER_QUERY,
        )
    ]
)

## Search result text and summary chain

RESULT_TEXT_MAX_CHARACTERS = 10000

search_result_text_and_summary_chain = (
    RunnableLambda(
        lambda x: {
            "search_result_text": web_scrape(url=x["result_url"])[
                :RESULT_TEXT_MAX_CHARACTERS
            ],
            "result_url": x["result_url"],
            "search_query": x["search_query"],
            "user_question": x["user_question"],
        }
    )
    | RunnableParallel(
        {
            "text_summary": SUMMARY_PROMPT_TEMPLATE | get_llm() | StrOutputParser(),
            "result_url": lambda x: x["result_url"],
            "user_question": lambda x: x["user_question"],
        }
    )
    | RunnableLambda(
        lambda x: {
            "summary": f"Source Url: {x['result_url']}\nSummary:{x['text_summary']}",
            "user_question": x["user_question"],
        }
    )
)

search_and_summarization_chain = (
    search_result_urls_chain
    | search_result_text_and_summary_chain.map()  # parallelize for each url
    | RunnableLambda(
        lambda x: {
            "summary": "\n".join([i["summary"] for i in x]),
            "user_question": x[0]["user_question"] if len(x) > 0 else "",
        }
    )
)

# Web Research chain

web_research_chain = (
    assistant_instructions_chain
    | web_searches_chain
    | search_and_summarization_chain.map()  # parallelize for each web search
    | RunnableLambda(
        lambda x: {
            "research_summary": "\n\n".join([i["summary"] for i in x]),
            "user_question": x[0]["user_question"] if len(x) > 0 else "",
        }
    )
    | RESEARCH_REPORT_PROMPT_TEMPLATE
    | get_llm()
    | StrOutputParser()
)
