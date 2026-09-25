"""Command-line entry point for the RAG pipeline.

Exposes the indexing, retrieval, generation and evaluation commands through
Python Fire. Run it as a module, e.g. ``python -m src index``.

Any uncaught exception is reported in red with its type, message and the
file and line where it was raised, and the process exits with status 1.
"""
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
    """Dispatch the CLI subcommand given in ``sys.argv`` via Python Fire.

    Available subcommands: ``index``, ``search``, ``search_dataset``,
    ``answer``, ``answer_dataset`` and ``evaluate``. Fire turns their
    keyword arguments into command-line flags (e.g. ``--k 5``).
    """
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
            f"{Colors.RED.value}[ERROR] - [{type(e).__name__}]\n"
            f"Error during the process...\n"
            f"Details: {e} (occurred in {tb.filename} at line {tb.lineno})"
            f"{Colors.RESET.value}\n"
        )
        sys.exit(1)
