from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel("gemini-1.5-flash")

agent = Agent(
  model=model,
  system_prompt="You are a helpful custom support agent. Be concise and friendly.",
)

response = agent.run_sync("How can I track my order #12345?")
print(response.data)
print(response.all_messages())