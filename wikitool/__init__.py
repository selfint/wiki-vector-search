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
        queries: Annotated[
            str | list[str], Doc("Query (or list of queries) to search.")
        ],
        top_k: Annotated[int, Doc("Amount of results to return.")] = 5,
        chunk_size: Annotated[
            int | None,
            Doc("Chunk size for [wikitool.llm_provider.LLMProvider.chunk][]."),
        ] = None,
        chunk_overlap: Annotated[
            int | None,
            Doc("Chunk overlap for [wikitool.llm_provider.LLMProvider.chunk][]."),
        ] = None,
    ) -> list[list[str]]:
        """
        Search the source using the llm provider.

        Returns:
            List of search results.

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
            >>> results = tool.search(["query_1", "query_2"], top_k, 512, 64)
            >>> assert isinstance(results, list)
            >>> assert len(results) == 2
            >>> assert isinstance(results[0][0], str)
            >>> assert len(results[0]) == top_k

            ```

        Note:
            Even when a single query is provided, a list of lists is returned.
            So, to get the result of a single query:

            ```pycon
            >>> from wikitool import WikiTool
            >>> from wikitool.extras.st_provider import STProvider
            >>> from wikitool.sources.wiki_source import WikiProvider
            >>> tool = WikiTool(
            ...         source=WikiProvider("WikiTool wikitool@test.com", "test"),
            ...         llm=STProvider("thenlper/gte-small"),
            ...        )
            >>> top_k = 3
            >>> results = tool.search("single query", top_k, 512, 64)
            >>> query_result = results[0]
            >>> assert isinstance(query_result, list)
            >>> assert isinstance(query_result[0], str)
            >>> assert len(query_result) == top_k

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

        hits = self._llm.search(query_embedding, corpus_embeddings, top_k)

        return [[chunks[h] for h in query_hits] for query_hits in hits]
