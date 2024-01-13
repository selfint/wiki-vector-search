import collections
from typing import TypeAlias, TypedDict

import requests
from wikipediaapi import Wikipedia, WikipediaPageSection


class ChunkMeta(TypedDict):
    titles: list[str]


class Chunk(TypedDict):
    text: str
    meta: ChunkMeta


class WikiProvider:
    def __init__(self, user_agent: str, language: str = "en") -> None:
        self._user_agent = user_agent

        self._endpoint = f"https://{language}.wikipedia.org/w/api.php"
        self._session = requests.Session()
        self._client = Wikipedia(user_agent, language)

    def search(self, query: str, max_pages: int = 1) -> list[Chunk]:
        titles = self._get_titles(query, max_pages)

        chunks = []
        for title in titles:
            page = self._client.page(title)
            tree = build_tree(page.sections)
            chunks.extend(build_tree_chunks(tree, [page.title]))

        return chunks

    def _get_titles(self, query: str, n: int = 10) -> list[str]:
        response = self._session.get(
            self._endpoint,
            headers={
                "User-Agent": self._user_agent,
            },
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "limit": n,
                "srsearch": query,
            },
        )

        assert response.ok, (response.status_code, response.text)

        output = response.json()["query"]["search"]

        titles = [hit["title"] for hit in output]

        return titles


_Tree: TypeAlias = dict[str, tuple[str, "_Tree"]]


def _tree() -> collections.defaultdict:
    return collections.defaultdict(_tree)


def build_tree(
    sections: list[WikipediaPageSection],
    tree: _Tree | None = None,
) -> _Tree:
    if tree is None:
        tree = _tree()

    for s in sections:
        tree[s.title] = (s.text, build_tree(s.sections))

    return dict(tree)


def build_text_chunk(text: str, titles: list[str]) -> Chunk:
    return {
        "text": text,
        "meta": {
            "titles": titles,
        },
    }


def build_tree_chunks(tree: _Tree, titles: list[str] | None = None) -> list[Chunk]:
    chunks = []
    titles = titles or []

    for k, (text, subtrees) in tree.items():
        chunks.append(build_text_chunk(text, titles + [k]))

        for subtitle, (subtext, subtree) in subtrees.items():
            chunks.append(build_text_chunk(subtext, titles + [k, subtitle]))
            chunks.extend(
                build_tree_chunks(
                    subtree,
                    titles,
                )
            )

    return chunks
