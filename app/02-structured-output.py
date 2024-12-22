from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel("gemini-1.5-flash")


"""
This example shows how to get structured, type-safe responses from the agent.

key concepts:

- Using pydantic models to define the response structured
- Type validation and safety
- Field description for better model understanding.
"""
class ResponseModel(BaseModel):
  """Structured response with metadata"""

  response: str
  needs_esclation: bool
  follow_up_required: bool
  sentiment: str = Field(description="Customer sentiment analysis")

agent = Agent(
  model=model,
  result_type=ResponseModel,
  system_prompt=(
    "You are an intelligent customr support agent."
    "Analyze queries carefully and provide structured responses."
  ),
)

response = agent.run_sync("How can I track my order #12345?")
print(response.data)
print(response.data.model_dump_json(indent=2))
