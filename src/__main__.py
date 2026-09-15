import sys
import traceback

import fire

from src.aux.colors import Colors
from src.commands import (
    answer,
    answer_dataset,
    evaluate,
    index,
    search,
    search_dataset,
)


def main() -> None:
    fire.Fire({
        'index': index,
        'search': search,
        'search_dataset': search_dataset,
        'answer': answer,
        'answer_dataset': answer_dataset,
        'evaluate': evaluate,
    })  # type: ignore[no-untyped-call]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        print(
            f"{Colors.RED.value}[ERROR] - "
            f"Error during the process...\n"
            f"Details: {e} (occurred in {tb.filename} at line {tb.lineno})"
            f"{Colors.RESET.value}\n"
        )
        sys.exit(1)
