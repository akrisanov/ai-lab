from research_engine_lcel import question, web_research_chain


def main():
    web_research_report = web_research_chain.invoke(question)
    print(web_research_report)


if __name__ == "__main__":
    main()
