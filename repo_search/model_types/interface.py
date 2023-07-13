from abc import ABC, abstractmethod
from typing import List

class ModelType(ABC):
    @abstractmethod
    def generate_embedding_for_chunk(self, chunk: str, verbose: bool = False) -> List[float]:
        pass
    
    @abstractmethod
    def get_max_chunk_length(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass