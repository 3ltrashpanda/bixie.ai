import os
from typing import List, Optional, Any

import openai
import chromadb
from chromadb.config import Settings
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIClient
from transformers import pipeline


class VulnAnalyzer:
    """
    Provides vulnerability analysis capabilities using embeddings, LLMs, and context storage.
    Integrates Semantic Kernel (OpenAI), HuggingFace, and ChromaDB for context/history.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        hf_model: str = "facebook/codebert-base",
        chroma_dir: str = ".chromadb",
    ):
        # OpenAI setup
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

        # Semantic Kernel setup
        self.kernel = Kernel()
        self.kernel.add_text_service(
            "openai",
            OpenAIClient(
                api_key=self.openai_api_key,
                organization=os.getenv("OPENAI_ORG_ID", None),
            ),
        )

        # HuggingFace pipeline for code embeddings or classification
        self.hf = pipeline("feature-extraction", model=hf_model, tokenizer=hf_model)

        # ChromaDB client for context storage
        self.chroma = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir)
        )
        self.collection = self._init_collection()

    def _init_collection(self) -> chromadb.api.Collection:
        # create or get collection for context
        try:
            return self.chroma.get_collection(name="vuln_context")
        except Exception:
            return self.chroma.create_collection(name="vuln_context")

    def embed_code(self, code_snippet: str) -> List[float]:
        """
        Generate embedding for a code snippet using HuggingFace model.
        """
        embeddings = self.hf(code_snippet)[0][0]
        return embeddings

    def store_context(self, doc_id: str, content: str, metadata: Optional[dict] = None):
        """
        Store a piece of context or history into ChromaDB.
        """
        emb = self.embed_code(content)
        self.collection.add(
            documents=[content],
            embeddings=[emb],
            ids=[doc_id],
            metadatas=[metadata or {}],
        )

    def retrieve_context(self, query: str, n_results: int = 5) -> List[Any]:
        """
        Retrieve top relevant contexts from ChromaDB for a given query.
        """
        q_emb = self.embed_code(query)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        return list(zip(results["documents"][0], results["metadatas"][0]))

    def analyze_binary(self, binary_bytes: bytes, prompt: str) -> str:
        """
        Use OpenAI via Semantic Kernel to analyze a binary blob for vulnerabilities.
        """
        # store binary as base64 if needed, retrieve context
        context = self.retrieve_context(prompt)
        prompt_with_ctx = """\
Using the following context: {ctx}
Analyze this binary (base64): {blob}
Provide vulnerabilities or suspicious patterns.""".format(
            ctx=context,
            blob=binary_bytes.hex(),
        )
        result = self.kernel.run(prompt_with_ctx, services=["openai"])
        return result

    def analyze_source(self, source_code: str) -> str:
        """
        Static analysis using OpenAI to find insecure patterns in source.
        """
        instruction = (
            "Identify insecure patterns, potential buffer overflows, and improper use of crypto in the following code:\n" + source_code
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=instruction,
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].text.strip()

    def sample_usage(self):
        """
        Example workflow:
        1. Store context
        2. Embed and retrieve
        3. Analyze binary and source
        """
        # 1. store some context
        self.store_context("ctx1", "Last analysis found unsafe memcpy usage.")

        # 2. retrieve
        ctx = self.retrieve_context("memcpy overflow")
        print("Retrieved Context:", ctx)

        # 3. analyze
        vuln_report = self.analyze_source(
            "char buf[10]; strcpy(buf, user_input);"
        )
        print("Vulnerability Report:", vuln_report)

