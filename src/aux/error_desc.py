from enum import Enum


class ErrorCodes(Enum):

    MAX_SIZE_CHUNK = "max_chunk_size needs to be less than 2001. " \
        "Applying default max value: 800."

    FILE_NOT_FOUND = " couldn't be found."

    PERMISSION = " couldn't be read/wrote. "\
        "Please, check the file/folder permissions. "

    OS_ERROR = " couldn't be found or read/write. "\
        "Please, check if the file exists and its permissions."

    CHUNK_SIZE_NOK = "The max_chunk_size value introduced is not a valid "\
        "integer."

    K_NOK = "The K value introduce is not a valid integer."

    QUERY_NOK = "The query introduced is not a valid string."

    DATASET_PATH_NOK = "The dataset path indicated is not a valid string or "\
        "does not exists."

    SAVE_FOLDER_NOK = "The save directory indicated is not a valid string."

    SAVE_FOLDER_EQ_DATASET = "Saving path and dataset path cannot be the"\
        " same. Otherwise, dataset would be overwritten."

    STUDENT_FILE_NOK = "The file indicated as student file is not a valid "\
        "string or does not exists."
