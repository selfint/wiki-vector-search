from typing import Annotated
from typing_extensions import Doc
from wikipediaapi import Wikipedia
from .llm_provider import LLMProvider

__version__ = "0.1.0"


class WikiTool:
    def __init__(
        self,
        client: Annotated[Wikipedia, Doc("Wikipedia client")],
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
        self._client = client
        self._llm = llm_provider
