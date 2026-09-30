"""User-facing error and warning messages."""
from enum import Enum


class ErrorCodes(Enum):
    """Message texts shown when a command fails or adjusts its input.

    Callers add the colour codes and the ``[ERROR]``/``[WARNING]`` prefix.
    The messages come in two forms:

    - Complete sentences, shown on their own (e.g. ``K_NOK``,
      ``QUERY_NOK``).
    - Fragments starting with a space (``FILE_NOT_FOUND``, ``PERMISSION``,
      ``OS_ERROR``), which are appended right after the offending file
      path, e.g. ``f"The file '{path}'{ErrorCodes.OS_ERROR.value}"``.

    ``MAX_SIZE_CHUNK`` is a warning rather than an error: the index is
    still built, with the default chunk size.
    """

    MAX_SIZE_CHUNK = "max_chunk_size needs to be less than 2001 and " \
        "greater than 200. Applying default max value: 800."

    FILE_NOT_FOUND = " couldn't be found."

    PERMISSION = " couldn't be read/wrote. "\
        "Please, check the file/folder permissions. "

    OS_ERROR = " couldn't be found or read/write. "\
        "Please, check if the file exists and its permissions."

    CHUNK_SIZE_NOK = "The max_chunk_size value introduced is not a valid "\
        "integer."

    K_NOK = "The K value introduce is not a valid positive integer."

    QUERY_NOK = "The query introduced is not a valid string."

    DATASET_PATH_NOK = "The dataset path indicated is not a valid string or "\
        "does not exists."

    SAVE_FOLDER_NOK = "The save directory indicated is not a valid string."

    SAVE_FOLDER_EQ_DATASET = "Saving path and dataset path cannot be the"\
        " same. Otherwise, dataset would be overwritten."

    STUDENT_FILE_NOK = "The file indicated as student file is not a valid "\
        "string or does not exists."

    STUDENT_DATASET_SAME = "The student file and dataset file indicated are "\
        "the same."

    BONUS_NOK = "The bonus value introduced is not a valid "\
        "boolean."
