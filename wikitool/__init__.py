from typing import Annotated

from typing_extensions import Doc

from .llm_provider import LLMProvider
from .sources.wiki_source import WikiProvider

__version__ = "0.1.0"


class WikiTool:
    def __init__(
        self,
        source: Annotated[WikiProvider, Doc("Source provider")],
        llm: Annotated[LLMProvider, Doc("LLM Provider")],
    ) -> None:
        """
        Wiki Tool.

        Example:
            Create a WikiTool object with Wikipedia and SentenceTransformers.

            ```pycon
            >>> from wikitool import WikiTool
            >>> from wikitool.extras.st_provider import STProvider
            >>> from wikitool.sources.wiki_source import WikiProvider
            >>> tool = WikiTool(
            ...         source=WikiProvider("WikiTool wikitool@test.com", "test"),
            ...         llm=STProvider("thenlper/gte-small"),
            ...        )

            ```
        """
        self._source = source
        self._llm = llm

    def search(
        self,
        queries: str | list[str],
        top_k: int = 5,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """
        Search the source using the llm provider.

        Example:
            Search Wikipedia using SentenceTransformers.

            ```pycon
            >>> from wikitool import WikiTool
            >>> from wikitool.extras.st_provider import STProvider
            >>> from wikitool.sources.wiki_source import WikiProvider
            >>> tool = WikiTool(
            ...         source=WikiProvider("WikiTool wikitool@test.com", "test"),
            ...         llm=STProvider("thenlper/gte-small"),
            ...        )
            >>> top_k = 3
            >>> results = tool.search("test", top_k, 512, 64)
            >>> assert isinstance(results, list)
            >>> assert isinstance(results[0], str)
            >>> assert len(results) == top_k

            ```
        """
        if isinstance(queries, str):
            queries = [queries]

        chunks = []
        for query in queries:
            results = self._source.search(query)
            for result in results:
                texts = self._llm.chunk(result["text"], chunk_size, chunk_overlap)
                chunks.extend(texts)

        corpus_embeddings = self._llm.embed_corpus(chunks)

        query_embedding = self._llm.embed_queries(queries)

        hits = self._llm.search(query_embedding, corpus_embeddings, top_k)[0]

        return [chunks[h] for h in hits]
