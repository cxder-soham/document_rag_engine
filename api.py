# api.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import time

from src import config
from src.document_loader import DocumentLoader
from src.embeddings import get_embedding_generator
from src.search_engine import create_search_engine
try:
    from src.augmented_generation import AugmentedGenerator
    augmented_generation_available = True
except Exception as e:
    logging.warning(f"Augmented generation not available: {str(e)}")
    augmented_generation_available = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_engine_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Document Retriever Search API",
    description="Search engine for document retrieval with RAG capabilities",
    version="1.0.0"
)

# Define request/response models
class SearchRequest(BaseModel):
    query: str
    raw_mode: bool = False
    top_k: Optional[int] = None

class SearchResult(BaseModel):
    title: str
    content: str
    category: str
    score: float
    document_id: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    augmented_response: Optional[str] = None
    processing_time: float

# Global variables to store initialized components
document_loader = None
embedding_generator = None
search_engine = None
augmented_generator = None

@app.on_event("startup")
async def initialize_components():
    """Initialize all components when the API starts up"""
    global document_loader, embedding_generator, search_engine, augmented_generator
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA is available with {device_count} device(s)")
        else:
            logger.info("CUDA is not available. Using CPU.")
            config.USE_GPU = False
        
        # Step 1: Load documents
        logger.info("Loading documents...")
        document_loader = DocumentLoader(
            categories=config.DATASET_CATEGORIES,
            max_documents=config.MAX_DOCUMENTS
        )
        documents = document_loader.load_newsgroups()
        
        # Step 2: Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedding_generator = get_embedding_generator()
        
        # Step 3: Generate document embeddings
        logger.info("Generating document embeddings...")
        document_embeddings = embedding_generator.generate_embeddings(documents)
        
        # Step 4: Create and build search index
        logger.info("Building search index...")
        search_engine = create_search_engine()
        search_engine.build_index(document_embeddings, document_loader)
        
        # Step 5: Initialize augmented generator if available
        if augmented_generation_available:
            try:
                logger.info("Initializing augmented generator...")
                augmented_generator = AugmentedGenerator(
                    model_name="gemini-2.0-flash",
                    max_length=config.GENERATION_MAX_LENGTH,
                    api_key=config.GEMINI_API_KEY
                )
                logger.info("Augmented generation initialized successfully!")
            except Exception as e:
                logger.warning(f"Failed to initialize augmented generation: {str(e)}")
        
        logger.info("API initialization complete!")
    
    except Exception as e:
        logger.exception("Error during API initialization")
        raise RuntimeError(f"Failed to initialize search components: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {
        "status": "online",
        "message": "Document Retriever Search API is running",
        "augmented_generation": augmented_generator is not None
    }

@app.get("/info")
async def get_info():
    """Get information about the loaded dataset"""
    if not document_loader:
        raise HTTPException(status_code=503, detail="Search engine not fully initialized")
    
    return {
        "total_documents": len(document_loader.loaded_documents),
        "categories": document_loader.category_names,
        "distribution": document_loader.get_category_distribution()
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a semantic search query on the document collection
    """
    if not all([document_loader, embedding_generator, search_engine]):
        raise HTTPException(status_code=503, detail="Search engine not fully initialized")
    
    start_time = time.time()
    
    try:
        # Generate query embedding
        query_embedding = embedding_generator.generate_query_embedding(request.query)
        
        # Perform search
        top_k = request.top_k if request.top_k is not None else config.TOP_K_RESULTS
        raw_results = search_engine.search(query_embedding, num_results=top_k)
        
        # Format results
        results = []
        for result in raw_results:
            # Convert SearchResult objects to API SearchResult models
            document = result.document
            results.append(SearchResult(
                title=document.metadata.get("title", "Untitled"),  # Use metadata dictionary
                content=document.content[:500] + "..." if len(document.content) > 500 else document.content,
                category=document.metadata["category_name"],  # Use correct metadata field
                score=float(result.score),
                document_id=str(document.metadata["id"])  # Convert ID to string and use metadata
            ))
        
        # Generate augmented response if enabled and not in raw mode
        augmented_response = None
        if augmented_generator and not request.raw_mode and results:
            try:
                documents_for_context = [
                    {
                        "id": str(r.document_id),
                        "content": r.content,
                        "category": r.category,
                        "metadata": {
                            "category_name": r.category,
                            "id": r.document_id
                        }
                    }
                    for r in results
                ]
                augmented_response = augmented_generator.generate_response(
                    request.query, documents_for_context
                )
            except Exception as e:
                logger.error(f"Error generating augmented response: {str(e)}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query=request.query,
            augmented_response=augmented_response,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.exception("Error processing search request")
        raise HTTPException(status_code=500, detail=f"Search processing error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)