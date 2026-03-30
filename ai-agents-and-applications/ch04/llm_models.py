from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        # LangChain automatically imports OPENAI_API_KEY and uses it for requests
        model_name="gpt-5-nano",
    )
