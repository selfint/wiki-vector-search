from wikitool import WikiTool


def test_wikitool_search(mock_wiki_provider, mock_llm_provider):
    tool = WikiTool(source=mock_wiki_provider, llm=mock_llm_provider)

    tool.search("search")
