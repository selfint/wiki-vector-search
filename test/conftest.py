from wikitool.llm_provider import LLMProvider
from wikitool.sources.wiki_source import WikiProvider
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm_provider() -> LLMProvider:
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def mock_wiki_provider() -> WikiProvider:
    return MagicMock(spec=WikiProvider)
