from typing import Annotated
from typing_extensions import Doc

__version__ = "0.1.0"


class WVS:
    """Wiki Vector Search"""

    def greet(
        self,
        name: Annotated[str, Doc("The name of the person to greet")],
    ) -> str:
        """
        Greet a name.

        **Example**
        ```python

        >>> import wiki_vector_search as wvs
        >>> wvs.WVS().greet("user")
        'Hello, User!'

        ```
        """

        return f"Hello, {name.lower().capitalize()}!"

    def world():
        print("hello")
