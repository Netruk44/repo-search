from abc import ABC, abstractmethod
from typing import List

class ModelType(ABC):
    @abstractmethod
    def generate_embedding_for_document(self, chunk: str, verbose: bool = False) -> List[float]:
        pass

    @abstractmethod
    def generate_embedding_for_query(self, chunk: str, verbose: bool = False) -> List[float]:
        pass
    
    @abstractmethod
    def get_max_document_chunk_length(self) -> int:
        pass

    @abstractmethod
    def get_max_query_chunk_length(self) -> int:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        pass