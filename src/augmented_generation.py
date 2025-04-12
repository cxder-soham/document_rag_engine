"""
Augmented Generation Module for the Document Retriever Search Engine

This module integrates with the existing search functionality to generate
answers based on retrieved documents using the gemini-2.0-flash model via
the google.genai API.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Union

from rich.console import Console

# Import the gemini API client from google.genai
from google import genai

# Import from existing project modules
from src.search_engine import SearchResult
from src import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class AugmentedGenerator:
    """
    A class that handles generating augmented responses based on retrieved documents
    using the gemini-2.0-flash model via the google.genai API.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        max_length: int = 512,
        api_key: Optional[str] = config.GEMINI_API_KEY
    ):
        """
        Initialize the AugmentedGenerator.
        
        Args:
            model_name: Name of the model to use.
            max_length: Maximum length of generated text (in tokens).
            api_key: API key for accessing the gemini-2.0-flash model via google.genai.
                     If not provided, the environment variable GEMINI_API_KEY will be used.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Determine API key: from argument or environment variable
        self.api_key = config.GEMINI_API_KEY #dont use -> os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("No API key provided for gemini-2.0-flash. "
                           "Please set the GEMINI_API_KEY environment variable or pass it as an argument.")
        
        # Initialize the google.genai client with the API key
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Successfully initialized google.genai client for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing google.genai client: {str(e)}")
            raise RuntimeError(f"Failed to initialize google.genai client: {str(e)}")
    
    def _format_context(self, results: List[SearchResult], max_docs: int = 5) -> str:
        """
        Format retrieved documents into a coherent context string.
        
        Args:
            results: List of search results.
            max_docs: Maximum number of documents to include in the context.
        
        Returns:
            Formatted context string.
        """
        # Sort results by score (highest first) and limit to max_docs
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:max_docs]
        
        context_parts = []
        for i, result in enumerate(sorted_results):
            doc_content = result.document.content.strip()
            category = result.document.metadata.get('category_name', 'Unknown')
            score = result.score
            doc_text = (
                f"DOCUMENT {i+1} [Relevance: {score:.2%}, Category: {category}]\n"
                f"{doc_content}\n"
            )
            context_parts.append(doc_text)
        return "\n---\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        results: List[SearchResult],
        max_docs: int = 5
    ) -> Dict[str, Union[str, float]]:
        """
        Generate an augmented response based on the query and retrieved documents.
        
        Args:
            query: User's original query.
            results: List of search results from the search engine.
            max_docs: Maximum number of documents to include in the context.
        
        Returns:
            Dictionary containing the generated response and metadata.
        """
        if not results:
            logger.warning("No documents provided for augmented generation")
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "confidence": 0.0
            }
        
        # Format the context from retrieved documents
        formatted_context = self._format_context(results, max_docs)
        
        # Create prompt for the model
        prompt = f"""Answer the following query based on the provided context. 

            QUERY: {query}

            CONTEXT:
            {formatted_context}

            ANSWER:"""
        
        logger.info("Sending request to gemini-2.0-flash model via google.genai API")
        try:
            # Call the API with the prompt and max token setting
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Retrieve generated text from the API response
            generated_text = response.text.strip()
            
            # Calculate an average confidence score based on retrieved document relevance
            if results:
                avg_score = sum(r.score for r in results[:max_docs]) / min(len(results), max_docs)
            else:
                avg_score = 0.0
            
            logger.info(f"Generated response with confidence score: {avg_score:.2f}")
            return {
                "response": generated_text,
                "confidence": avg_score
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I encountered an error while generating a response. Please try again.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def display_augmented_response(
        self,
        query: str,
        results: List[SearchResult],
        max_docs: int = 5
    ) -> None:
        """
        Generate and display the augmented response.
        
        Args:
            query: User's original query.
            results: List of search results.
            max_docs: Maximum number of documents to include.
        """
        console.print("\n[bold cyan]Generating augmented response...[/bold cyan]")
        response_data = self.generate_response(query, results, max_docs)
        console.print("\n[bold green]Augmented Response:[/bold green]")
        console.print(response_data["response"])
        confidence = response_data.get("confidence", 0.0)
        console.print(f"\n[dim]Response confidence: {confidence:.2%}[/dim]")
        if results:
            used_docs = min(len(results), max_docs)
            console.print(f"[dim]Based on {used_docs} documents[/dim]")
