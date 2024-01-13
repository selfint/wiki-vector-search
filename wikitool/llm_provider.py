from typing import Protocol, TypeVar

T = TypeVar("T")


class LLMProvider(Protocol[T]):
    def embed_queries(self, texts: list[str] | str) -> T:
        ...

    def embed_corpus(self, texts: list[str] | str) -> T:
        ...

    def chunk(
        self,
        header: str,
        text: str,
        size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        ...

    def search(
        self,
        query_embeddings: T,
        corpus_embeddings: T,
        top_k: int,
    ) -> list[list[int]]:
        ...
