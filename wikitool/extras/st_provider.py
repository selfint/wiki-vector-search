from typing import TypeAlias

import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..llm_provider import LLMProvider

T: TypeAlias = Tensor | NDArray


class STProvider(LLMProvider[T]):
    def __init__(
        self,
        model: str,
        instruction: str | None = None,
        device: str | None = None,
        half: bool = False,
    ) -> None:
        self._device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self._model = SentenceTransformer(model, device=self._device)
        if half:
            self._model = self._model.half()
        self._model.compile()

        self._tokenizer = AutoTokenizer.from_pretrained(model)

        self._instruction = instruction

        self._normalize = True

    def embed_corpus(self, texts: list[str] | str) -> T:
        return self._model.encode(
            texts,
            device=self._device,
            normalize_embeddings=self._normalize,
            convert_to_numpy=self._device == "cpu",
            convert_to_tensor=self._device != "cpu",
        )  # type: ignore

    def embed_queries(self, texts: list[str] | str) -> T:
        if self._instruction is not None:
            texts = [self._instruction + t for t in texts]

        return self._model.encode(
            texts,
            device=self._device,
            normalize_embeddings=self._normalize,
            convert_to_numpy=self._device == "cpu",
            convert_to_tensor=self._device != "cpu",
        )  # type: ignore

    def chunk(self, text: str, size: int, overlap: int) -> list[str]:
        if len(text) <= 1:
            return [text]

        tokenizer: PreTrainedTokenizerFast = self._pipe.tokenizer  # type: ignore

        tokens = tokenizer.tokenize(
            text,
            max_length=size,
            stride=overlap,
            truncation=True,
            add_special_tokens=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        token_offsets: list[list[tuple[int, int]]] = tokens["offset_mapping"]  # type: ignore
        chunk_offsets = [(offsets[0][0], offsets[-1][1]) for offsets in token_offsets]

        chunks = [text[start:end] for start, end in chunk_offsets]

        return chunks

    def search(
        self,
        query_embeddings: T,
        corpus_embeddings: T,
        top_k: int,
    ) -> list[list[int]]:
        hits = semantic_search(
            query_embeddings=query_embeddings,  # type: ignore
            corpus_embeddings=corpus_embeddings,  # type: ignore
            top_k=top_k,
        )

        return [[h["corpus_id"] for h in query_hits] for query_hits in hits]
