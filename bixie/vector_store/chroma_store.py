import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Dict, Any, Optional
import hashlib
import os

class ChromaStore:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed explicitly.")

        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=model_name
        )

        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.PersistentClient()

        self.collections = {}  # collection_name -> collection object
        self.collection_metadata = {}  # collection_name -> metadata
        self.current_collection = None

    def get_or_create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Returns the collection object, creating it if it doesn't exist.
        Optionally sets metadata if creating.
        """
        if collection_name not in self.collections:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.collections[collection_name] = collection
            if metadata is not None:
                self.collection_metadata[collection_name] = metadata
        return self.collections[collection_name]

    def create_collection(self, collection_name: str, metadata: Dict[str, Any]) -> None:
        collection = self.get_or_create_collection(collection_name, metadata)
        self.current_collection = collection

    def switch_collection(self, collection_name: str) -> None:
        collection = self.get_or_create_collection(collection_name)
        self.current_collection = collection

    def get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        return self.collection_metadata.get(collection_name)

    @staticmethod
    def generate_id(text: str, metadata: Dict[str, Any]) -> str:
        blob = text + str(metadata.get("file", ""))
        return hashlib.sha256(blob.encode()).hexdigest()

    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        if not self.current_collection:
            raise RuntimeError("No collection selected. Use create_collection or switch_collection first.")
        vector_id = self.generate_id(text, metadata)
        self.current_collection.add(
            ids=[vector_id],
            documents=[text],
            metadatas=[metadata]
        )
        return vector_id
    def query_across_collections(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search all known collections and return the top-k most relevant matches across all of them.
        """
        results = []
        for name, collection in self.collections.items():
            try:
                query_result = collection.query(
                    query_texts=[query_text],
                    n_results=top_k
                )
                for i in range(len(query_result["ids"][0])):
                    results.append({
                        "collection": name,
                        "id": query_result["ids"][0][i],
                        "distance": query_result["distances"][0][i],
                        "metadata": query_result["metadatas"][0][i],
                        "document": query_result["documents"][0][i],
                    })
            except Exception as e:
                print(f"[WARN] Failed to query collection '{name}': {e}")

        # Sort across collections by distance (ascending)
        return sorted(results, key=lambda x: x["distance"])[:top_k]

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.current_collection:
            raise RuntimeError("No collection selected. Use create_collection or switch_collection first.")
        results = self.current_collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        matches = []
        for i in range(len(results["ids"][0])):
            match = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            matches.append(match)
        return matches

    #def query_all(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]: 
    #    results = self.memory.query_across_collections(prompt, top_k=3)
    #    if results:
    #        context_blocks = [
    #            f"[{item['collection']}] {item['document']}" for item in similar_contexts
    #        ]
    #        context = "\n\n[RELATED CONTEXT]\n" + "\n---\n".join(context_blocks)
    #        #print(context)
    #    return context

    def get_document(self, vector_id: str) -> Optional[Dict[str, Any]]:
        if not self.current_collection:
            raise RuntimeError("No collection selected. Use create_collection or switch_collection first.")
        results = self.current_collection.get(ids=[vector_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0], "metadata": results["metadatas"][0]
                }
        return None

    def delete_document(self, vector_id: str) -> None:
        if not self.current_collection:
            raise RuntimeError("No collection selected. Use create_collection or switch_collection first.")
        self.current_collection.delete(ids=[vector_id])

