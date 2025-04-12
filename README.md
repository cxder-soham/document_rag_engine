# Document Retriever Search Engine with LangChain

A semantic search engine built with LangChain, PyTorch, and SentenceTransformers that allows users to search through the 20 newsgroups dataset using natural language queries.

## Features

- Loads and processes documents from the 20 newsgroups dataset
- Generates document embeddings using state-of-the-art transformer models
- Efficiently utilizes GPU acceleration (optimized for NVIDIA RTX 4070)
- Performs semantic search to retrieve the most relevant documents for a given query
- Windows-compatible implementation (no faiss-gpu dependency)

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- LangChain
- SentenceTransformers
- scikit-learn (for loading the 20 newsgroups dataset)
- Annoy (for vector indexing, Windows-compatible)

## Project Structure

```
project_root/
  ├── README.md
  ├── requirements.txt
  ├── main.py
  └── src/
      ├── __init__.py
      ├── document_loader.py
      ├── embeddings.py
      ├── search_engine.py
      └── config.py
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/document-retriever-search-engine.git
cd document-retriever-search-engine
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script to start the search engine:
```bash
python main.py
```

2. Enter your search queries when prompted.

3. To exit the program, type 'exit', 'quit', or press Ctrl+C.

## How it Works

1. The application loads documents from the 20 newsgroups dataset.
2. Documents are processed and transformed into embeddings using a SentenceTransformer model.
3. The embeddings are indexed using the Annoy library (a fast, cross-platform vector search library).
4. When a query is entered, it is also transformed into an embedding and compared against the document embeddings.
5. The most semantically similar documents are retrieved and displayed.

## GPU Acceleration

The application automatically detects and utilizes available CUDA-capable GPUs. If you have an NVIDIA RTX 4070, the embedding generation process will be significantly accelerated.

## Customization

You can modify the configuration settings in the `src/config.py` file:
- Change the embedding model
- Adjust the number of documents to load
- Modify the number of search results to display
- Configure other parameters related to the search engine

## Troubleshooting

If you encounter issues with GPU acceleration:
1. Ensure you have installed PyTorch with CUDA support. You can verify this by running:
```python
import torch
print(torch.cuda.is_available())
```
2. Make sure your NVIDIA drivers are up to date.
3. Check that your GPU is properly recognized by PyTorch:
```python
import torch
print(torch.cuda.get_device_name(0))
```

## License

[MIT License](LICENSE)