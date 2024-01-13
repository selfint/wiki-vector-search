# Quickstart

Get started right away with [SentenceTransformers](#st):

```sh
$ pip install https://github.com/selfint/wikitool/releases/download/0.1.0/wikitool-0.1.0-py3-none-any.whl
$ pip install 'wikitool[st]'
```

And then search:

```pycon
>>> from wikitool import WikiTool
>>> from wikitool.extras.st_provider import STProvider
>>> from wikitool.sources.wiki_source import WikiProvider
>>> tool = WikiTool(
...         # change 'test' to your language
...         source=WikiProvider("WikiTool wikitool@test.com", "test"),
...         llm=STProvider("thenlper/gte-small"),
...        )
>>> top_k = 3
>>> chunk_size = 512
>>> chunk_overlap = 64
>>> results = tool.search(["query_1", "query_2"], top_k, chunk_size, chunk_overlap)

```

## Installation

Requires Python 3.10+.

### Python 3.12

[PyTorch doesn't support Python 3.12](https://download.pytorch.org/whl/torch/) as of Jan 2024.
So the [wikitool.extras.st_provider.STProvider][] won't work with Python 3.12.

Install a specific version:

```sh
$ version="0.1.0"
$ pip install https://github.com/selfint/wikitool/releases/download/$version/wikitool-$version-py3-none-any.whl
```

Or install directly from the source on GitHub:

```sh
$ pip install git+https://github.com/selfint/wikitool
```

### Extras

Extras can be installed **after** [installing the package](#installation), like so:

```sh
$ pip install 'wikitool[extra]'
```

#### `st`

Provides a [wikitool.llm_provider.LLMProvider][] implementation is provided using [sentence_transformers.SentenceTransformer][].

## Search Wikipedia

See the [search examples][wikitool.WikiTool.search] in the [Code Reference](reference).
