"""
Document loader module for the Document Retriever Search Engine.

This module is responsible for loading and preprocessing the 20 newsgroups dataset.
It provides functions to fetch documents and convert them into a format suitable
for embedding generation and indexing.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.progress import Progress, track
from sklearn.datasets import fetch_20newsgroups

from . import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up rich console for better output formatting
console = Console()


@dataclass
class Document:
    """
    Class representing a document with its content and metadata.
    
    Attributes:
        content (str): The text content of the document
        metadata (dict): Additional information about the document
    """
    content: str
    metadata: Dict


class DocumentLoader:
    """
    Class for loading and preprocessing documents from the 20 newsgroups dataset.
    """
    
    def __init__(self, categories: Optional[List[str]] = None, max_documents: Optional[int] = None):
        """
        Initialize the DocumentLoader.
        
        Args:
            categories: Optional list of categories to load (None loads all categories)
            max_documents: Maximum number of documents to load (None loads all documents)
        """
        self.categories = categories
        self.max_documents = max_documents
        self.loaded_documents = []
        self.category_names = []
        
    def load_newsgroups(self) -> List[Document]:
        """
        Load documents from the 20 newsgroups dataset.
        
        Returns:
            List[Document]: A list of Document objects
        """
        console.print("[bold cyan]Loading 20 newsgroups dataset...[/bold cyan]")
        
        # Load the 20 newsgroups dataset
        dataset = fetch_20newsgroups(
            subset='all',
            categories=self.categories,
            shuffle=True,
            random_state=config.RANDOM_STATE,
            remove=('headers', 'footers', 'quotes'),  # Remove headers, footers, and quotes to focus on content
            download_if_missing=True
        )
        
        self.category_names = dataset.target_names
        
        # Log the number of categories and documents
        logger.info(f"Loaded {len(self.category_names)} categories")
        logger.info(f"Total documents available: {len(dataset.data)}")
        
        # Limit the number of documents if specified
        data = dataset.data
        targets = dataset.target
        if self.max_documents and self.max_documents < len(data):
            data = data[:self.max_documents]
            targets = targets[:self.max_documents]
            logger.info(f"Limited to {len(data)} documents as specified")
        
        # Create Document objects with content and metadata
        documents = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing documents...", total=len(data))
            
            for i, (text, target) in enumerate(zip(data, targets)):
                # Clean the text (remove extra whitespace, normalize newlines, etc.)
                cleaned_text = self._clean_text(text)
                
                # Create metadata for the document
                metadata = {
                    'id': i,
                    'category_id': int(target),
                    'category_name': dataset.target_names[target]
                }
                
                # Add the document to our list
                documents.append(Document(content=cleaned_text, metadata=metadata))
                progress.update(task, advance=1)
        
        self.loaded_documents = documents
        console.print(f"[bold green]Successfully loaded {len(documents)} documents from "
                      f"{len(self.category_names)} categories.[/bold green]")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: The raw text content
            
        Returns:
            str: Cleaned text
        """
        # Convert to string in case we get anything else
        text = str(text)
        
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        
        # Ensure the text is not too long or too short
        if len(text) <= 10:
            text = "Empty or very short document."
        
        return text
    
    def get_document_by_id(self, doc_id: int) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: The ID of the document to retrieve
            
        Returns:
            Optional[Document]: The document if found, None otherwise
        """
        for doc in self.loaded_documents:
            if doc.metadata['id'] == doc_id:
                return doc
        return None
    
    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of documents across categories.
        
        Returns:
            Dict[str, int]: Dictionary mapping category names to document counts
        """
        distribution = {}
        for doc in self.loaded_documents:
            category = doc.metadata['category_name']
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def print_sample_documents(self, n: int = 3) -> None:
        """
        Print sample documents to show what's been loaded.
        
        Args:
            n: Number of sample documents to print
        """
        if not self.loaded_documents:
            console.print("[yellow]No documents loaded yet.[/yellow]")
            return
        
        n = min(n, len(self.loaded_documents))
        console.print(f"\n[bold]Sample of {n} documents:[/bold]")
        
        for i in range(n):
            doc = self.loaded_documents[i]
            console.print(f"\n[bold cyan]Document {i+1}[/bold cyan]")
            console.print(f"[bold]Category:[/bold] {doc.metadata['category_name']}")
            # Print just the first 200 characters of content for brevity
            preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            console.print(f"[bold]Content (preview):[/bold] {preview}")


def load_documents() -> List[Document]:
    """
    Helper function to load documents using the default configuration.
    
    Returns:
        List[Document]: A list of loaded documents
    """
    loader = DocumentLoader(
        categories=config.DATASET_CATEGORIES,
        max_documents=config.MAX_DOCUMENTS
    )
    return loader.load_newsgroups()


if __name__ == "__main__":
    # Example usage
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")