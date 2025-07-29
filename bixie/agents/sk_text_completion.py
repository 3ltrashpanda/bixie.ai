import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.memory.chroma import ChromaMemoryStore

class SKAgent:
    def __init__(
        self,
        use_openai: bool = True,
        use_hf: bool = True,
        use_memory: bool = False,
        chroma_persist_dir: str = "./chroma_store",
    ):
        # 1) Create the kernel
        self.kernel = Kernel()

        # 2) Wire up text completion providers
        if use_openai:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY in env")
            self.kernel.add_text_completion_service(
                "openai",
                OpenAITextCompletion(
                    deployment="gpt-3.5-turbo",  # free-tier model
                    api_key=key,
                    api_base="https://api.openai.com/v1",
                ),
            )

        if use_hf:
            token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not token:
                raise ValueError("Set HUGGINGFACE_API_TOKEN in env")
            self.kernel.add_text_completion_service(
                "hf",
                HuggingFaceTextCompletion(
                    model_id="gpt2-xl",  # or your preferred HF model
                    api_token=token,
                ),
            )

        # 3) Setup memory (Chroma-backed)
        if use_memory:
            store = ChromaMemoryStore(persist_directory=chroma_persist_dir)
            self.memory = SemanticTextMemory(store)
            # register memory with a name, e.g. "buffer"
            self.kernel.register_memory(self.memory, name="buffer")
        else:
            self.memory = None

        # 4) (Optional) Load/define your skills here
        #    e.g. self.kernel.import_directory("my_skills", "MySkills")

    def run(
        self,
        skill_name: str,
        prompt: str,
        provider: str = "openai",
        max_tokens: int = 512,
    ) -> str:
        """
        Invoke a registered semantic skill.
        skill_name: name of skill/function
        prompt: input text
        provider: "openai" or "hf"
        """
        if provider not in self.kernel.ai_services:
            raise ValueError(f"Provider {provider} not configured")

        # Optionally save to memory
        if self.memory:
            self.memory.save_user_input(prompt)

        # Run the skill
        result = (
            self.kernel.run(prompt, skill_name, completion_service=provider)
        )

        # Optionally save to memory
        if self.memory:
            self.memory.save_assistant_response(result)

        return result

# === Usage ===
if __name__ == "__main__":
    # ensure env vars are set: OPENAI_API_KEY, HUGGINGFACE_API_TOKEN
    agent = SKAgent(use_openai=True, use_hf=True, use_memory=True)
    out = agent.run("SummarizeSkill", "Explain how TLS handshake works.", provider="hf")
    print(out)

