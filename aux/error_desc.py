from enum import Enum

class ErrorCodes(Enum):
    MAX_SIZE_CHUNK = "max_chunk_size needs to be less than 2001. " \
        "Applying default max value: 2000."

    FILE_NOT_FOUND = " couldn't be found. Please check it. " \
        "Exitting..."

    PERMISSION = " couldn't be opened. Please, check the file permissions. " \
        "Exitting..."
