from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
