"""
Augmented Generation Module for the Document Retriever Search Engine

This module integrates with the existing search functionality to generate
answers based on retrieved documents using the gemini-2.0-flash model.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Union
from google import genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# Import from existing project modules
from src.search_engine import SearchResult

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
    using the gemini-2.0-flash model.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemini-2.0-flash",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Initialize the AugmentedGenerator.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing AugmentedGenerator with {model_name} on {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None
            )
            
            # Move model to the appropriate device
            if self.device == 'cuda':
                # Optimize for NVIDIA RTX 4070
                if torch.cuda.get_device_properties(0).total_memory >= 12 * 1024 * 1024 * 1024:  # 12 GB
                    logger.info("Using GPU memory optimization for RTX 4070")
                else:
                    logger.warning("GPU detected but may not have enough memory for optimal performance")
            
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def _format_context(self, results: List[SearchResult], max_docs: int = 5) -> str:
        """
        Format retrieved documents into a coherent context string.
        
        Args:
            results: List of search results
            max_docs: Maximum number of documents to include in the context
        
        Returns:
            Formatted context string
        """
        # Sort results by score (highest first) and limit to max_docs
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:max_docs]
        
        context_parts = []
        for i, result in enumerate(sorted_results):
            # Extract document content and metadata
            doc_content = result.document.content.strip()
            category = result.document.metadata.get('category_name', 'Unknown')
            score = result.score
            
            # Format document with metadata
            doc_text = (
                f"DOCUMENT {i+1} [Relevance: {score:.2%}, Category: {category}]\n"
                f"{doc_content}\n"
            )
            context_parts.append(doc_text)
        
        # Join all documents with separators
        return "\n---\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        results: List[SearchResult],
        max_docs: int = 5,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[str, float]]:
        """
        Generate an augmented response based on the query and retrieved documents.
        
        Args:
            query: User's original query
            results: List of search results from the search engine
            max_docs: Maximum number of documents to include in the context
            generation_params: Optional parameters to override default generation settings
        
        Returns:
            Dictionary containing the generated response and metadata
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
        prompt = f"""
        Answer the following query based on the provided context. If the context doesn't contain relevant information, state that you don't have enough information to answer accurately.
        
        QUERY: {query}
        
        CONTEXT:
        {formatted_context}
        
        ANSWER:
        """
        
        # Prepare generation parameters
        params = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Override with custom parameters if provided
        if generation_params:
            params.update(generation_params)
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Log token usage for monitoring
            input_tokens = len(inputs.input_ids[0])
            logger.info(f"Input prompt length: {input_tokens} tokens")
            
            # Generate the response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **params
                )
            
            # Decode the generated text
            # We want only the new tokens, not the input prompt
            response_text = self.tokenizer.decode(
                generated_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate a confidence score based on average relevance of documents
            if results:
                avg_score = sum(r.score for r in results[:max_docs]) / min(len(results), max_docs)
            else:
                avg_score = 0.0
                
            logger.info(f"Generated response with confidence score: {avg_score:.2f}")
            
            return {
                "response": response_text.strip(),
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
        Generate and display the augmented response in a formatted way.
        
        Args:
            query: User's original query
            results: List of search results
            max_docs: Maximum number of documents to use
        """
        console.print("\n[bold cyan]Generating augmented response...[/bold cyan]")
        
        response_data = self.generate_response(query, results, max_docs)
        
        # Display the generated response with formatting
        console.print("\n[bold green]Augmented Response:[/bold green]")
        console.print(response_data["response"])
        
        # Show confidence information
        confidence = response_data.get("confidence", 0.0)
        console.print(f"\n[dim]Response confidence: {confidence:.2%}[/dim]")
        
        # Display info about source documents
        if results:
            used_docs = min(len(results), max_docs)
            console.print(f"[dim]Based on {used_docs} documents[/dim]")