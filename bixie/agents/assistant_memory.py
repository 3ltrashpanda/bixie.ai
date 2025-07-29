# chroma_memory_wrapper.py

from typing import Optional, List
from bixie.vector_store.chroma_store import ChromaStore  # Your existing class

class ChromaMemoryWrapper:
    def __init__(
        self,
        chroma_store: ChromaStore,
        collection_name: str,
        metadata: Optional[dict] = None,
        top_k: int = 3,
    ):
        self.store = chroma_store
        self.collection_name = collection_name
        self.top_k = top_k
        self.store.get_or_create_collection(collection_name, metadata)
        self.store.switch_collection(collection_name)

    def inject_context(self, prompt: str) -> str:
        """Retrieve context and prepend it to the prompt"""
        matches = self.store.query_across_collections(prompt, top_k=self.top_k)
        if not matches:
            return prompt

        context_blocks = [
            f"[{m['collection']}] {m['document']}" for m in matches
        ]
        context = "\n\n[RELATED CONTEXT]\n" + "\n---\n".join(context_blocks)
        return f"{context}\n\n{prompt}"

    def save_exchange(self, prompt: str, response: str, role: str = "user") -> None:
        """Save the prompt/response into memory"""
        self.store.add_document(prompt, {"role": role})
        self.store.add_document(response, {"role": "assistant"})

