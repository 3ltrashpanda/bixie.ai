import asyncio
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentRegistry, ChatHistoryAgentThread
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# Define a plugin with kernel functions
class MenuPlugin:
    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"

# YAML spec for the agent
AGENT_YAML = """
type: chat_completion_agent
name: Assistant
description: A helpful assistant.
instructions: Answer the user's questions using the menu functions.
tools:
  - id: MenuPlugin.get_specials
    type: function
  - id: MenuPlugin.get_item_price
    type: function
model:
  options:
    temperature: 0.7
"""

USER_INPUTS = [
    "Hello",
    "What is the special soup?",
    "What does that cost?",
    "Thank you",
]

async def main():
    kernel = Kernel()
    kernel.add_plugin(MenuPlugin(), plugin_name="MenuPlugin")

    agent: ChatCompletionAgent = await AgentRegistry.create_from_yaml(
        AGENT_YAML, kernel=kernel, service=OpenAIChatCompletion()
    )

    thread: ChatHistoryAgentThread | None = None

    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        response = await agent.get_response(user_input, thread=thread)
        print(f"# {response.name}: {response}")
        thread = response.thread

    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())
