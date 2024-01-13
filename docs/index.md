# Quickstart

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

## Search Wikipedia

See the [search examples][wikitool.WikiTool.search] in the [Code Reference](reference).
