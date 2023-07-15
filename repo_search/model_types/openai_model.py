import openai
import os
import tiktoken
import time
from typing import List, Optional

from .interface import ModelType

class OpenAIModel(ModelType):
    def __init__(
            self, 
            model_name: Optional[str] = None, 
            max_chunk_length: Optional[int] = None,
            encoder_name: Optional[str] = None):
        
        self.model_name = model_name if model_name is not None else 'text-embedding-ada-002'
        self.max_chunk_length = max_chunk_length if max_chunk_length is not None else 8191
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        assert self.api_key, "OPENAI_API_KEY environment variable must be set in order to use the OpenAI model type."
        openai.api_key = self.api_key

        self.encoder_name = encoder_name if encoder_name is not None else 'cl100k_base'
        self.encoder = tiktoken.get_encoding(self.encoder_name)

    def generate_embedding_for_document(self, chunk: str, verbose: bool = False) -> List[float]:
        return self.generate_embedding_for_chunk(chunk, verbose)

    def generate_embedding_for_query(self, chunk: str, verbose: bool = False) -> List[float]:
        return self.generate_embedding_for_chunk(chunk, verbose)

    def generate_embedding_for_chunk(self, chunk: str, verbose: bool = False) -> List[float]:
        encoded_chunk = self.encoder.encode(chunk)
        assert len(encoded_chunk) <= self.max_chunk_length, f'Chunk length {len(chunk)} exceeds max chunk length {self.max_chunk_length}.'

        # OpenAI API is flaky, use retry with exponential backoff.
        current_try = 0
        max_tries = 5

        while current_try <= max_tries:
            current_try += 1

            try:
                embedding_response = openai.Embedding.create(
                    input=chunk,
                    model=self.model_name,
                )
                break
            except openai.error.OpenAIError as e:
                if verbose:
                    print(f'WARNING: OpenAI API error: {e}')
                
                if current_try == max_tries:
                    raise e
            
            # Exponential backoff
            time.sleep(2**current_try)
        
        return embedding_response['data'][0]['embedding']
    
    def get_max_document_chunk_length(self) -> int:
        return self.max_chunk_length
    
    def get_max_query_chunk_length(self) -> int:
        return self.max_chunk_length
    
    def tokenize(self, text: str) -> List[int]:
        return self.encoder.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)
    
    @staticmethod
    def get_model_type() -> str:
        return 'openai'

    def get_model_name(self) -> str:
        return self.model_name