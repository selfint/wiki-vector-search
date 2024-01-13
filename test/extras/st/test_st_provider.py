from functools import lru_cache

import pytest

from wikitool.extras.st_provider import STProvider
from wikitool.llm_provider import LLMProvider


@lru_cache(maxsize=1)
def _st_provider() -> STProvider:
    """Only load model once"""

    return STProvider("thenlper/gte-small", device="cpu")


@pytest.fixture
def st_provider() -> LLMProvider:
    return _st_provider()


def test_embed(st_provider, snapshot):
    embedding = st_provider.embed_corpus(["this is a test", "and another test"])

    # snapshot first 2 digits to bypass float issues
    snap = [[int(a * 100) for a in b] for b in embedding]

    assert snap == snapshot


def test_chunk(st_provider, snapshot):
    text = "The quick brown fox jumps over the lazy dog"

    chunks = st_provider.chunk(text, 3, 1)

    assert chunks == snapshot


def test_search(st_provider, snapshot):
    corpus = [
        "The name of the dog is Kevin",
        "The house is blue",
        "It is Tuesday",
    ]

    queries = [
        "What is the pet's name?",
        "What color is the house?",
        "What day is it?",
    ]

    top_k = 1
    results = st_provider.search(
        st_provider.embed_queries(queries),
        st_provider.embed_corpus(corpus),
        top_k=top_k,
    )

    assert [len(r) for r in results] == [top_k] * len(results)
    results = [[corpus[i] for i in r] for r in results]

    assert list(zip(queries, results)) == snapshot
