from abc import ABC, abstractmethod

class RAGSystem(ABC):

    def __init__(self, embedding_tool):
        self.embedding_tool = embedding_tool
    
    """Abstract base class for RAG systems."""
    @abstractmethod
    def search(self, query: str) -> list[dict]:
        """Abstract search method to be implemented by subclasses."""
        pass
