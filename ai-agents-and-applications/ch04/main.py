from research_engine_lcel import question, web_research_chain


def main():
    for chunk in web_research_chain.stream(question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
