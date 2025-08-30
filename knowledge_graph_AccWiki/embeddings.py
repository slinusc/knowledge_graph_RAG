#!/usr/bin/env python3
"""
Reusable embedding model class for both ingestion and query processes.

This module provides a unified interface for creating embeddings using
SentenceTransformers, with automatic device detection and configuration.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np

# Default configuration
DEFAULT_MODEL_NAME = "thellert/accphysbert_cased"
DEFAULT_USE_CUDA = "true"  # "true", "false", "auto"


class EmbeddingModel:
    """
    A reusable embedding model class that handles device detection,
    model loading, and embedding creation for both ingestion and queries.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_cuda: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            use_cuda: CUDA usage preference ("true", "false", "auto")
            hf_token: Hugging Face authentication token if needed
        """
        self.model_name = model_name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL_NAME)
        self.use_cuda = use_cuda or os.environ.get("USE_CUDA", DEFAULT_USE_CUDA).lower()
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        self._model = None
        self._device = None
        self._vector_dim = None
        
    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def device(self) -> str:
        """Get the device being used by the model."""
        if self._device is None:
            self._determine_device()
        return self._device
    
    @property
    def vector_dim(self) -> int:
        """Get the vector dimensions of the model."""
        if self._vector_dim is None:
            # For accphysbert_cased (BERT-base), dimension is 768
            if "bert" in self.model_name.lower():
                self._vector_dim = 768
            else:
                # For other models, we'd need to load and check
                # This is a fallback that requires model loading
                test_embedding = self.encode(["test"], show_progress_bar=False)
                self._vector_dim = len(test_embedding[0])
        return self._vector_dim
    
    def _determine_device(self):
        """Determine which device to use for the model."""
        try:
            import torch
            
            if self.use_cuda == "false":
                self._device = "cpu"
            elif self.use_cuda == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:  # "true" (default)
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    logging.warning("CUDA requested but not available, falling back to CPU")
                    self._device = "cpu"
        except ImportError:
            logging.warning("PyTorch not available, using CPU")
            self._device = "cpu"
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine device
            if self._device is None:
                self._determine_device()
            
            logging.info(f"Loading embedding model: {self.model_name} on device: {self._device}")
            
            # Handle Hugging Face authentication if needed
            if self.hf_token:
                logging.info("Using Hugging Face authentication token")
                self._model = SentenceTransformer(
                    self.model_name, 
                    device=self._device, 
                    token=self.hf_token
                )
            else:
                self._model = SentenceTransformer(self.model_name, device=self._device)
            
            # Log GPU info if using CUDA
            if self._device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
        except ImportError as e:
            if "torch" in str(e):
                raise ImportError("PyTorch not installed. Install with: pip install torch")
            else:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]], 
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
        prefix: str = "passage"
    ) -> np.ndarray:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: Single text or list of texts to embed
            normalize_embeddings: Whether to normalize the embeddings
            batch_size: Batch size for processing
            convert_to_numpy: Whether to convert to numpy array
            show_progress_bar: Whether to show progress bar
            prefix: Prefix to add to texts (e.g., "passage", "query")
            
        Returns:
            Numpy array of embeddings
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Add prefix if specified
        if prefix:
            prefixed_texts = [f"{prefix}: {text}" for text in texts]
        else:
            prefixed_texts = texts
        
        # Create embeddings
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar
        )
        
        return embeddings
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """
        Create embeddings for a query text.
        
        Args:
            query: Query text to embed
            **kwargs: Additional arguments passed to encode()
            
        Returns:
            Numpy array of embeddings
        """
        # Set query-specific defaults
        kwargs.setdefault('prefix', 'query')
        kwargs.setdefault('show_progress_bar', False)
        
        return self.encode(query, **kwargs)
    
    def encode_passages(self, passages: List[str], **kwargs) -> np.ndarray:
        """
        Create embeddings for passage texts (for ingestion).
        
        Args:
            passages: List of passage texts to embed
            **kwargs: Additional arguments passed to encode()
            
        Returns:
            Numpy array of embeddings
        """
        # Set passage-specific defaults
        kwargs.setdefault('prefix', 'passage')
        kwargs.setdefault('show_progress_bar', True)
        
        return self.encode(passages, **kwargs)


# Global instance for easy reuse
_global_embedder = None


def get_embedder(
    model_name: Optional[str] = None,
    use_cuda: Optional[str] = None,
    hf_token: Optional[str] = None
) -> EmbeddingModel:
    """
    Get a global embedding model instance (singleton pattern).
    
    Args:
        model_name: Model name (only used on first call)
        use_cuda: CUDA preference (only used on first call)
        hf_token: HF token (only used on first call)
        
    Returns:
        EmbeddingModel instance
    """
    global _global_embedder
    
    if _global_embedder is None:
        _global_embedder = EmbeddingModel(
            model_name=model_name,
            use_cuda=use_cuda,
            hf_token=hf_token
        )
    
    return _global_embedder


def reset_embedder():
    """Reset the global embedder instance."""
    global _global_embedder
    _global_embedder = None


if __name__ == "__main__":
    embedder = get_embedder()
    # Test encoding
    test_text = "Hello, world!"
    embedding = embedder.encode(test_text)
    print("Embedding:", embedding)