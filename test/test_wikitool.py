from wikitool import WikiTool


def test_wiki_tool(mock_wiki, mock_llm_provider, snapshot):
    tool = WikiTool(
        source=mock_wiki,
        llm=mock_llm_provider,
    )

    results = tool.search("search")

    assert results == snapshot
