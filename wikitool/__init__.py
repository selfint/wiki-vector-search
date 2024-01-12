from typing import Annotated
from typing_extensions import Doc
from wikipediaapi import Wikipedia
from .llm_provider import LLMProvider
from .sources.wiki_source import WikiProvider
from functools import lru_cache

__version__ = "0.1.0"


class WikiTool:
    def __init__(
        self,
        source: Annotated[WikiProvider, Doc("Wikipedia client")],
        llm: Annotated[LLMProvider, Doc("LLM Provider")],
    ) -> None:
        """
        Wiki Tool.

        Example:
            Create a WikiTool object
            ```pycon
            >>> from wikipediaapi import Wikipedia
            >>> from wikitool import WikiTool
            >>> from wikitool.extras.st_provider import STProvider
            >>> tool = WikiTool(
            ...         client=Wikipedia("user_agent"),
            ...         llm=STProvider("thenlper/gte-small"),
            ...        )

            ```
        """
        self._source = source
        self._llm = llm

    def search(self, query: str, top_k: int = 5) -> list[str]:
        chunks = self._source.search(query)

        corpus = [c["text"] for c in chunks]
        corpus_embeddings = self._llm.embed_corpus(corpus)

        query_embedding = self._llm.embed_queries(query)

        hits = self._llm.search(query_embedding, corpus_embeddings, top_k)[0]

        return [corpus[h] for h in hits]
