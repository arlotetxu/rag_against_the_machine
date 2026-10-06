from enum import Enum


class Colors(str, Enum):
    """ANSI escape codes used to color terminal output.

    Wrap a message between a color and ``RESET`` so the color does not
    leak into later output. Use ``.value`` inside f-strings: from Python
    3.12 on, formatting a member of a ``str`` mixin enum gives its name
    (``Colors.RED``) instead of the escape code.

    The first members are the standard 8 colors plus ``BOLD``; the ones
    from ``PURPLE`` on use the 256-color palette (``38;5;<n>``), which
    some terminals do not support.
    """

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    PURPLE = "\033[38;5;128m"
    BROWN = "\033[38;5;130m"
    ORANGE = "\033[38;5;208m"
    MAROON = "\033[38;5;52m"
    GOLD = "\033[38;5;220m"
    DARKRED = "\033[38;5;88m"
    CRIMSON = "\033[38;5;197m"
    VIOLET = "\033[38;5;93m"
