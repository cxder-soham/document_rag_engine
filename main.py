#!/usr/bin/env python3
"""
Document Retriever Search Engine with LangChain

This is the main script that ties together all components of the Document Retriever 
Search Engine. It demonstrates how to load documents, generate embeddings, build 
a search index, and handle user queries.

Usage:
    python main.py
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel

from src import config
from src.document_loader import DocumentLoader, load_documents
from src.embeddings import EmbeddingGenerator, get_embedding_generator
from src.search_engine import SearchEngine, create_search_engine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up rich console
console = Console()


def check_cuda_availability() -> None:
    """
    Check if CUDA is available and print GPU information.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        console.print(f"[bold green]CUDA is available with {device_count} device(s)[/bold green]")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
            console.print(f"[green]Device {i}: {device_name} with {total_memory:.2f} GB memory[/green]")
            
            # Add specific info for RTX 4070
            if "RTX 4070" in device_name:
                console.print("[bold cyan]NVIDIA RTX 4070 detected - performance optimizations will be applied[/bold cyan]")
    else:
        console.print("[yellow]CUDA is not available. Using CPU for computations.[/yellow]")
        # Automatically disable GPU usage in config if CUDA is not available
        config.USE_GPU = False


def print_welcome_message() -> None:
    """
    Print a welcome message with project information.
    """
    welcome_text = """
    Document Retriever Search Engine with LangChain
    ----------------------------------------------
    
    This application allows you to perform semantic searches on the 20 newsgroups dataset.
    The search uses deep learning models to understand the meaning of your queries and
    find the most relevant documents.
    
    - Type your search query to find relevant documents
    - Type 'info' to see dataset information
    - Type 'exit' or 'quit' to exit the program
    """
    
    console.print(Panel(welcome_text, title="Welcome", border_style="cyan", expand=False))


def print_dataset_info(document_loader: DocumentLoader) -> None:
    """
    Print information about the loaded dataset.
    
    Args:
        document_loader: The DocumentLoader instance with loaded documents
    """
    if not document_loader or not document_loader.loaded_documents:
        console.print("[yellow]No documents loaded yet.[/yellow]")
        return
    
    # Get category distribution
    distribution = document_loader.get_category_distribution()
    
    # Print dataset info
    console.print("\n[bold cyan]Dataset Information[/bold cyan]")
    console.print(f"Total documents: [bold]{len(document_loader.loaded_documents)}[/bold]")
    console.print(f"Number of categories: [bold]{len(document_loader.category_names)}[/bold]")
    
    # Print category distribution
    console.print("\n[bold cyan]Category Distribution:[/bold cyan]")
    for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(document_loader.loaded_documents)) * 100
        console.print(f"- {category}: [bold]{count}[/bold] documents ({percentage:.1f}%)")
    
    # Print sample documents
    document_loader.print_sample_documents(n=2)


def main() -> None:
    """
    Main function to run the Document Retriever Search Engine with Augmented Generation.
    """
    try:
        # Start timing for performance measurement
        start_time = time.time()
        
        # Check for CUDA availability
        check_cuda_availability()
        
        # Print welcome message
        print_welcome_message()
        
        # Step 1: Load documents
        console.print("\n[bold]Step 1: Loading documents...[/bold]")
        document_loader = DocumentLoader(
            categories=config.DATASET_CATEGORIES,
            max_documents=config.MAX_DOCUMENTS
        )
        documents = document_loader.load_newsgroups()
        
        # Step 2: Initialize embedding generator
        console.print("\n[bold]Step 2: Initializing embedding generator...[/bold]")
        embedding_generator = get_embedding_generator()
        
        # Step 3: Generate document embeddings
        console.print("\n[bold]Step 3: Generating document embeddings...[/bold]")
        document_embeddings = embedding_generator.generate_embeddings(documents)
        
        # Step 4: Create and build search index
        console.print("\n[bold]Step 4: Building search index...[/bold]")
        search_engine = create_search_engine()
        search_engine.build_index(document_embeddings, document_loader)
        
        # Step 5: Initialize augmented generator
        console.print("\n[bold]Step 5: Initializing augmented generator...[/bold]")
        try:
            from src.augmented_generation import AugmentedGenerator
            augmented_generator = AugmentedGenerator(
                model_name="gemini-2.0-flash",
                max_length=config.GENERATION_MAX_LENGTH,
                api_key=config.GEMINI_API_KEY
            )
            augmented_generation_enabled = True
            console.print("[green]Augmented generation initialized successfully![/green]")
        except Exception as e:
            logger.warning(f"Failed to initialize augmented generation: {str(e)}")
            console.print("[yellow]Augmented generation could not be initialized. Running in search-only mode.[/yellow]")
            augmented_generation_enabled = False
        
        # Calculate and display initialization time
        init_time = time.time() - start_time
        console.print(f"\n[bold green]Initialization completed in {init_time:.2f} seconds[/bold green]")
        
        # Step 6: Interactive search loop
        console.print("\n[bold cyan]Ready for search queries. Type 'exit' or 'quit' to exit.[/bold cyan]")
        if augmented_generation_enabled:
            console.print("[cyan]Type 'raw' before your query to see only search results without augmented generation.[/cyan]")
        
        while True:
            # Get query from user
            query = console.input("\n[bold]Enter your search query:[/bold] ")
            
            # Check for exit command
            if query.lower() in ['exit', 'quit']:
                console.print("[bold cyan]Exiting search engine. Goodbye![/bold cyan]")
                break
            
            # Check for info command
            if query.lower() == 'info':
                print_dataset_info(document_loader)
                continue
            
            # Skip empty queries
            if not query.strip():
                continue
            
            # Check if user wants raw search results only
            raw_mode = False
            if query.lower().startswith('raw '):
                raw_mode = True
                query = query[4:].strip()  # Remove the 'raw ' prefix
            
            # Process query
            query_start_time = time.time()
            
            # Generate query embedding
            query_embedding = embedding_generator.generate_query_embedding(query)
            
            # Perform search
            results = search_engine.search(query_embedding)
            
            # Calculate query time
            query_time = time.time() - query_start_time
            
            # Display results
            search_engine.display_search_results(query, results)
            console.print(f"[dim]Query processed in {query_time:.4f} seconds[/dim]")
            
            # Generate augmented response if enabled and not in raw mode
            if augmented_generation_enabled and not raw_mode and results:
                try:
                    # Start timing for augmented generation
                    gen_start_time = time.time()
                    
                    # Generate and display augmented response
                    augmented_generator.display_augmented_response(query, results)
                    
                    # Calculate and display generation time
                    gen_time = time.time() - gen_start_time
                    console.print(f"[dim]Augmented response generated in {gen_time:.4f} seconds[/dim]")
                except Exception as e:
                    logger.error(f"Error generating augmented response: {str(e)}")
                    console.print("[yellow]Failed to generate augmented response. See logs for details.[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Search engine interrupted. Exiting gracefully.[/bold cyan]")
    except Exception as e:
        logger.exception("An error occurred during execution")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print("[bold red]Please check the log file for more details.[/bold red]")
    finally:
        # Cleanup if needed
        pass


if __name__ == "__main__":
    main()