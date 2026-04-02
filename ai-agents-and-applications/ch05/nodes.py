from llm_models import get_llm
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
    RELEVANCE_EVALUATION_PROMPT,
    RESEARCH_REPORT_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    WEB_SEARCH_PROMPT_TEMPLATE,
)
from utils import to_obj
from web_scraping import web_scrape
from web_searching import web_search

NUM_SEARCH_QUERIES = 3
NUM_SEARCH_RESULTS_PER_QUERY = 3
RESULT_TEXT_MAX_CHARACTERS = 10000


def parse_assistant_info(content: str) -> dict:
    result = to_obj(content)
    if isinstance(result, dict) and "assistant_instructions" in result:
        return result
    return {
        "assistant_type": "General research assistant",
        "assistant_instructions": content,
        "user_question": "",
    }


def parse_search_queries(content: str) -> list:
    result = to_obj(content)
    if isinstance(result, list):
        return result
    return []


def select_assistant(state: dict) -> dict:
    """Select the appropriate research assistant."""
    user_question = state["user_question"]
    prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=user_question)
    response = get_llm().invoke(prompt)
    assistant_info = parse_assistant_info(str(response.content))
    return {"assistant_info": assistant_info}


def generate_search_queries(state: dict) -> dict:
    """Generate search queries based on the question."""
    assistant_info = state["assistant_info"]
    user_question = state["user_question"]

    prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
        assistant_instructions=assistant_info["assistant_instructions"],
        user_question=user_question,
        num_search_queries=NUM_SEARCH_QUERIES,
    )
    response = get_llm().invoke(prompt)
    search_queries = parse_search_queries(str(response.content))
    return {"search_queries": search_queries}


def perform_web_searches(state: dict) -> dict:
    """Perform web searches for each generated search query."""
    search_queries = state["search_queries"]
    user_question = state["user_question"]

    search_results = []
    for query_obj in search_queries:
        urls = web_search(
            query=query_obj["search_query"],
            num_results=NUM_SEARCH_RESULTS_PER_QUERY,
        )
        for url in urls:
            search_results.append(
                {
                    "result_url": url,
                    "search_query": query_obj["search_query"],
                    "user_question": user_question,
                }
            )
    return {"search_results": search_results}


def summarize_search_results(state: dict) -> dict:
    """Scrape each search result URL and summarize its content."""
    search_results = state["search_results"]

    search_summaries = []
    for result in search_results:
        page_text = web_scrape(url=result["result_url"])[:RESULT_TEXT_MAX_CHARACTERS]
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            search_result_text=page_text,
            search_query=result["search_query"],
        )
        response = get_llm().invoke(prompt)
        summary = f"Source Url: {result['result_url']}\nSummary:{response.content}"
        search_summaries.append(
            {
                "summary": summary,
                "result_url": result["result_url"],
                "user_question": result["user_question"],
            }
        )

    research_summary = "\n\n".join([s["summary"] for s in search_summaries])
    return {"search_summaries": search_summaries, "research_summary": research_summary}


def evaluate_search_relevance(state: dict) -> dict:
    """Evaluate whether the gathered research is relevant to the question."""
    research_summary = state.get("research_summary", "")
    user_question = state["user_question"]

    prompt = RELEVANCE_EVALUATION_PROMPT.format(
        user_question=user_question,
        research_summary=research_summary[:3000],
    )
    response = get_llm().invoke(prompt)
    evaluation = to_obj(response.content)

    should_regenerate = not evaluation.get("is_relevant", True)
    iteration_count = state.get("iteration_count", 0) + 1
    return {
        "relevance_evaluation": evaluation,
        "should_regenerate_queries": should_regenerate,
        "iteration_count": iteration_count,
    }


def write_research_report(state: dict) -> dict:
    """Write the final research report from the gathered summaries."""
    research_summary = state.get("research_summary", "")
    user_question = state["user_question"]

    prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
        research_summary=research_summary,
        user_question=user_question,
    )
    response = get_llm().invoke(prompt)
    return {"final_report": response.content}
