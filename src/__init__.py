"""
Document Retriever Search Engine package.

This package contains modules for loading documents, generating embeddings,
and performing semantic search operations.
"""

# Import the main components to make them accessible when importing the package
from . import config
from . import document_loader
from . import embeddings
from . import search_engine

__version__ = "1.0.0"