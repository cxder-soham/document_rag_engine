"""
Embeddings module for the Document Retriever Search Engine.

This module is responsible for generating embeddings from documents using
a pre-trained language model. It handles GPU allocation and provides utilities
for working with the embeddings.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config
from .document_loader import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up rich console
console = Console()


class EmbeddingGenerator:
    """
    Class for generating embeddings from documents using a pre-trained model.
    It handles GPU allocation and provides utilities for working with embeddings.
    """
    
    def __init__(self, model_name: str = None, use_gpu: bool = None, gpu_device: int = None):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: Name of the SentenceTransformers model to use
            use_gpu: Whether to use GPU if available
            gpu_device: GPU device index to use
        """
        # Use config values as defaults if not provided
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_gpu = config.USE_GPU if use_gpu is None else use_gpu
        self.gpu_device = gpu_device or config.GPU_DEVICE
        
        self.device = self._get_device()
        self.model = self._load_model()
        
        # Update vector dimension in config based on model output
        config.VECTOR_DIM = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {config.VECTOR_DIM}")
    
    def _get_device(self) -> torch.device:
        """
        Determine the appropriate device (CPU or GPU) based on availability and settings.
        
        Returns:
            torch.device: The device to use for model inference
        """
        if self.use_gpu and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if self.gpu_device < device_count:
                device = torch.device(f"cuda:{self.gpu_device}")
                gpu_name = torch.cuda.get_device_name(self.gpu_device)
                logger.info(f"Using GPU: {gpu_name} (Device {self.gpu_device})")
                gpu_props = torch.cuda.get_device_properties(self.gpu_device)
                total_memory = gpu_props.total_memory / 1024**3  # Convert to GB
                logger.info(f"GPU Memory: {total_memory:.2f} GB")
                if "RTX 4070" in gpu_name:
                    logger.info("Detected NVIDIA RTX 4070. Optimizations are in place.")
                return device
            else:
                logger.warning(f"Requested GPU device {self.gpu_device} but only {device_count} devices are available. "
                               f"Falling back to device 0.")
                return torch.device("cuda:0")
        elif self.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        else:
            logger.info("Using CPU for embeddings (GPU usage is disabled)")
            return torch.device("cpu")
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the SentenceTransformer model and move it to the appropriate device.
        
        Returns:
            SentenceTransformer: The loaded model
        """
        console.print(f"[bold cyan]Loading embedding model: {self.model_name}[/bold cyan]")
        try:
            # Load the model and explicitly move it to the selected device
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded model: {self.model_name}")
            if self.device.type == 'cuda':
                try:
                    model = torch.jit.script(model)
                    logger.info("Applied TorchScript optimization")
                except Exception as e:
                    logger.warning(f"Could not apply TorchScript optimization: {e}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            console.print("[bold red]Failed to load embedding model. Check if the model name is correct.[/bold red]")
            raise
    
    def generate_embeddings(self, documents: List[Document], 
                            batch_size: int = 32) -> Dict[int, torch.Tensor]:
        """
        Generate embeddings for a list of documents on GPU.
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for embedding generation
            
        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping document IDs to GPU tensors
        """
        if not documents:
            logger.warning("No documents provided for embedding generation")
            return {}
        
        texts = [doc.content for doc in documents]
        doc_ids = [doc.metadata['id'] for doc in documents]
        
        console.print(f"[bold cyan]Generating embeddings for {len(texts)} documents...[/bold cyan]")
        embeddings = {}
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating embeddings...", total=len(texts))
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_ids = doc_ids[i:i+batch_size]
                # Generate embeddings as GPU tensors
                batch_embeddings = self.model.encode(batch_texts, 
                                                     show_progress_bar=False,
                                                     convert_to_tensor=True)
                for doc_id, embedding in zip(batch_ids, batch_embeddings):
                    embeddings[doc_id] = embedding
                progress.update(task, advance=len(batch_texts))
        
        console.print(f"[bold green]Successfully generated embeddings for {len(embeddings)} documents.[/bold green]")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> torch.Tensor:
        """
        Generate an embedding for a search query on GPU.
        
        Args:
            query: The search query text
            
        Returns:
            torch.Tensor: The query embedding on GPU
        """
        with torch.no_grad():
            embedding = self.model.encode(query, convert_to_tensor=True)
        return embedding


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Helper function to create an EmbeddingGenerator with default configuration.
    
    Returns:
        EmbeddingGenerator: Configured embedding generator instance
    """
    return EmbeddingGenerator(
        model_name=config.EMBEDDING_MODEL,
        use_gpu=config.USE_GPU,
        gpu_device=config.GPU_DEVICE
    )


if __name__ == "__main__":
    from .document_loader import load_documents
    documents = load_documents()
    generator = get_embedding_generator()
    doc_embeddings = generator.generate_embeddings(documents[:10])
    print(f"Generated {len(doc_embeddings)} embeddings")
    query_embedding = generator.generate_query_embedding("How to install Linux?")
    print(f"Query embedding shape: {query_embedding.shape}")
