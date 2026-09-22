"""crag -- controlled RAG chunking study.

Import-light on purpose: nothing here pulls in torch, transformers, datasets
or faiss, so the pure-Python half (tokenization, chunkers, metrics, retrieve,
stats) can be imported and tested on a machine with no GPU and no network.
"""

__version__ = "0.1.0"
