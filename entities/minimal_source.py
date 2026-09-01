from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class IndexedChunk(BaseModel):
    text = str
    metadata = MinimalSource