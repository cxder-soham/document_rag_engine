"""
Search engine module for the Document Retriever Search Engine.

This module handles indexing document embeddings and performing semantic search 
using the Annoy library for efficient similarity search on Windows systems.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from annoy import AnnoyIndex
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from . import config
from .document_loader import Document, DocumentLoader
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up rich console
console = Console()


@dataclass
class SearchResult:
    """
    Class representing a search result, including the document and its relevance score.
    
    Attributes:
        document (Document): The retrieved document
        score (float): The relevance score (higher is better)
    """
    document: Document
    score: float


class SearchEngine:
    """
    Class for indexing documents and performing semantic searches using Annoy.
    """
    
    def __init__(self, 
                vector_dim: int = None, 
                metric: str = 'angular', 
                n_trees: int = None,
                search_k: int = None,
                index_path: str = None):
        """
        Initialize the SearchEngine.
        
        Args:
            vector_dim: Dimension of the embedding vectors
            metric: Distance metric to use ('angular', 'euclidean', 'manhattan', 'hamming', 'dot')
            n_trees: Number of trees for the Annoy index (affects build time and accuracy)
            search_k: Number of nodes to inspect during search (affects search speed and accuracy)
            index_path: Path to save/load the Annoy index
        """
        # Use config values as defaults if not provided
        self.vector_dim = vector_dim or config.VECTOR_DIM
        self.n_trees = n_trees or config.N_TREES
        self.search_k = search_k or config.SEARCH_K
        self.index_path = index_path or config.ANNOY_INDEX_PATH
        
        self.metric = metric
        self.document_loader = None
        self.index = None
        self.id_to_index = {}  # Maps document IDs to Annoy index positions
        self.index_to_id = {}  # Maps Annoy index positions to document IDs
    
    def build_index(self, 
                   document_embeddings: Dict[int, np.ndarray], 
                   document_loader: DocumentLoader,
                   save_index: bool = True) -> None:
        """
        Build the search index from document embeddings.
        
        Args:
            document_embeddings: Dictionary mapping document IDs to embeddings
            document_loader: DocumentLoader instance with loaded documents
            save_index: Whether to save the index to disk
        """
        if not document_embeddings:
            logger.error("No document embeddings provided for indexing")
            return
        
        if not self.vector_dim:
            # Try to infer dimension from the first embedding
            first_embedding = next(iter(document_embeddings.values()))
            self.vector_dim = len(first_embedding)
            logger.info(f"Inferred vector dimension: {self.vector_dim}")
        
        # Initialize Annoy index
        self.index = AnnoyIndex(self.vector_dim, self.metric)
        
        # Store the document loader
        self.document_loader = document_loader
        
        # Add vectors to the index
        console.print(f"[bold cyan]Building search index with {len(document_embeddings)} documents...[/bold cyan]")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Building index...", total=len(document_embeddings))
            
            for i, (doc_id, embedding) in enumerate(document_embeddings.items()):
                self.index.add_item(i, embedding)
                self.id_to_index[doc_id] = i
                self.index_to_id[i] = doc_id
                progress.update(task, advance=1)
        
        # Build the index
        console.print(f"[bold cyan]Building trees (n_trees={self.n_trees})...[/bold cyan]")
        self.index.build(self.n_trees)
        
        console.print(f"[bold green]Search index built successfully with {len(document_embeddings)} documents.[/bold green]")
        
        # Save the index if requested
        if save_index:
            self._save_index()
    
    def _save_index(self) -> None:
        try:
            self.index.save(self.index_path)
            logger.info(f"Index saved to {self.index_path}")

            # Save mapping dicts
            meta_path = self.index_path + ".meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump((self.id_to_index, self.index_to_id), f)
            logger.info(f"Index metadata saved to {meta_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self, document_loader: DocumentLoader) -> bool:
        """
        Load a previously saved Annoy index from disk.
        
        Args:
            document_loader: DocumentLoader instance with loaded documents
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if not os.path.exists(self.index_path):
            logger.error(f"Index file not found: {self.index_path}")
            return False
        
        if not self.vector_dim:
            logger.error("Vector dimension must be set before loading an index")
            return False
        
        try:
            self.index = AnnoyIndex(self.vector_dim, self.metric)
            self.index.load(self.index_path)
            self.document_loader = document_loader

            # Load mapping metadata
            meta_path = self.index_path + ".meta.pkl"
            with open(meta_path, "rb") as f:
                self.id_to_index, self.index_to_id = pickle.load(f)

            logger.info(f"Successfully loaded index and metadata from {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, 
              query_embedding: np.ndarray, 
              num_results: int = None) -> List[SearchResult]:
        """
        Perform a semantic search using a query embedding.
        
        Args:
            query_embedding: The embedding of the search query
            num_results: Number of results to return
            
        Returns:
            List[SearchResult]: Ranked list of search results
        """
        if not self.index:
            logger.error("Search index has not been built or loaded")
            return []
        
        if not self.document_loader:
            logger.error("Document loader not available")
            return []
        
        # Use default from config if not specified
        num_results = num_results or config.NUM_SEARCH_RESULTS
        
        # Perform the search
        indices, distances = self.index.get_nns_by_vector(
            query_embedding, 
            num_results, 
            search_k=self.search_k, 
            include_distances=True
        )
        
        # Convert distances to similarity scores (1 - distance for angular distance)
        # This makes higher scores better, which is more intuitive
        if self.metric == 'angular':
            scores = [1 - distance for distance in distances]
        else:
            # For other metrics, just use inverse of distance (higher = better)
            scores = [1 / (1 + distance) for distance in distances]
        
        # Generate search results
        results = []
        for idx, score in zip(indices, scores):
            doc_id = self.index_to_id.get(idx)
            if doc_id is not None:
                document = self.document_loader.get_document_by_id(doc_id)
                if document:
                    results.append(SearchResult(document=document, score=score))
        
        return results
    
    def display_search_results(self, query: str, results: List[SearchResult]) -> None:
        """
        Display search results in a formatted table.
        
        Args:
            query: The original search query
            results: List of search results to display
        """
        if not results:
            console.print("[yellow]No matching documents found.[/yellow]")
            return
        
        console.print(f"\n[bold]Search results for query:[/bold] '{query}'")
        
        # Create a table for displaying results
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Rank", justify="center", style="cyan", width=5)
        table.add_column("Score", justify="center", style="green", width=10)
        table.add_column("Category", style="magenta", width=20)
        table.add_column("Content Preview", style="white")
        
        # Add rows for each result
        for i, result in enumerate(results):
            # Format the score as a percentage
            score_display = f"{result.score:.2%}"
            
            # Get a preview of the content (first 100 characters)
            content_preview = result.document.content[:100]
            if len(result.document.content) > 100:
                content_preview += "..."
            
            # Add the row to the table
            table.add_row(
                str(i + 1),
                score_display,
                result.document.metadata['category_name'],
                content_preview
            )
        
        console.print(table)


def create_search_engine() -> SearchEngine:
    """
    Helper function to create a SearchEngine with default configuration.
    
    Returns:
        SearchEngine: Configured search engine instance
    """
    return SearchEngine(
        vector_dim=config.VECTOR_DIM,
        n_trees=config.N_TREES,
        search_k=config.SEARCH_K,
        index_path=config.ANNOY_INDEX_PATH
    )


if __name__ == "__main__":
    # Example usage
    from .document_loader import load_documents
    from .embeddings import get_embedding_generator
    
    # Load documents
    document_loader = DocumentLoader()
    documents = document_loader.load_newsgroups()
    
    # Generate embeddings
    generator = get_embedding_generator()
    document_embeddings = generator.generate_embeddings(documents[:100])  # Just use 100 docs for testing
    
    # Build search index
    engine = create_search_engine()
    engine.build_index(document_embeddings, document_loader)
    
    # Example search
    query = "How to install Linux?"
    query_embedding = generator.generate_query_embedding(query)
    results = engine.search(query_embedding)
    
    # Display results
    engine.display_search_results(query, results)