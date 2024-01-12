from typing import Callable, TypeAlias
from transformers import PreTrainedTokenizerFast
import collections
from functools import partial


Tree: TypeAlias = dict[str, tuple[str, "Tree"]]


def _tree() -> collections.defaultdict:
    return collections.defaultdict(_tree)


def build_tree(sections, tree: Tree | None = None) -> Tree:
    if tree is None:
        tree = _tree()

    for s in sections:
        tree[s.title] = (s.text, build_tree(s.sections))

    return dict(tree)


def chunk_tokens(
    tokens: list[str],
    chunk_size: int,
    chunk_overlap: int,
    is_subword: Callable[[str], bool],
) -> list[list[str]]:
    chunks = []

    while len(tokens) > 0:
        # reduce chunk size to not end with subword
        chunk_len = chunk_size
        while (chunk_len + 1 < len(tokens)) and is_subword(tokens[chunk_len + 1]):
            chunk_len -= 1
        assert chunk_len > 0, "got empty chunk"

        chunk_tokens = tokens[:chunk_len]
        chunks.append(chunk_tokens)

        new_start = chunk_len
        if new_start > len(tokens):
            break

        # try to reduce overlap to not cut subwords
        overlap = chunk_overlap if chunk_overlap < chunk_len else 0
        while is_subword(tokens[new_start - overlap]):
            overlap -= 1
            if overlap == 0:
                print(
                    "failed to prevent subword cut:",
                    tokens[new_start - 1],
                    tokens[new_start],
                )
                break

        tokens = tokens[new_start - overlap :]

    return chunks


def build_text_chunks(
    text: str,
    title: str,
    title_path: list[str],
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[dict]:
    # build header
    header = (
        "\n".join(f"{'#' * (i + 1)} {t}" for i, t in enumerate(title_path + [title]))
        + "\n"
    )

    # split tokens into chunk sized chunks
    text_chunk_size = chunk_size - len(tokenizer.tokenize(header))
    text_chunks_tokens = chunk_tokens(
        tokenizer.tokenize(text),
        text_chunk_size,
        chunk_overlap,
        is_subword=lambda t: not t.startswith("▁"),
    )

    # build chunks
    chunks = [
        {
            "text": header + tokenizer.decoder.decode(text_chunk),
            "meta": {"title": title, "title_path": title_path},
        }
        for text_chunk in text_chunks_tokens
    ]

    return chunks


def build_tree_chunks(
    tree: Tree,
    title_path: list[str],
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[dict]:
    assert chunk_size > chunk_overlap * 2, f"overlap must be < {chunk_size // 2=}"
    _build_text_chunks = partial(
        build_text_chunks,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    title_path = title_path or []

    for k, (text, subtrees) in tree.items():
        chunks += _build_text_chunks(text, k, title_path)

        # build subtree chunks
        for subtitle, (subtext, subtree) in subtrees.items():
            chunks += _build_text_chunks(subtext, subtitle, title_path + [k])
            chunks += build_tree_chunks(
                subtree,
                title_path,
                tokenizer,
                chunk_size,
                chunk_overlap,
            )

    return chunks
