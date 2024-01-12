from wikitool.llm_provider import LLMProvider
import pytest
from unittest.mock import MagicMock
from wikipediaapi import Wikipedia


@pytest.fixture
def mock_llm_provider() -> LLMProvider:
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def mock_wiki() -> Wikipedia:
    return MagicMock(spec=Wikipedia)
