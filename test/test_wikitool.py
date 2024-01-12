from wikitool import WikiTool


def test_wiki_tool(mock_wiki, mock_llm_provider):
    tool = WikiTool(mock_wiki, mock_llm_provider)
