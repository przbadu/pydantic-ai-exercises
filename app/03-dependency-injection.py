from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
# from pydantic_ai.models.openai import OpenAIModel
from typing import List, Optional

# Fix the utils.markdown import error
import sys
sys.path.append("app")

from utils.markdown import to_markdown

model = GeminiModel("gemini-1.5-flash")


"""
This example shows how to  use dependencies and context in agents.

key concepts:
- Defining complex data models with pydantic
- Injecting runtime dependencies into the agent
- Using dynamic system prompts based on the context
"""

class Order(BaseModel):
  """Structure for Order Details"""
  order_id: str
  status: str
  items: List[str]


class CustomerDetails(BaseModel):
  """Structure for incoming Customer queries"""
  customer_id: str
  name: str
  email: str
  orders: Optional[List[Order]] = None


class ResponseModel(BaseModel):
  """Structured response with metadata"""
  response: str
  needs_escalation: bool
  follow_up_required: bool
  sentiment: str = Field(description="Customer sentiment analysis")


agent = Agent(
  model=model,
  result_type=ResponseModel,
  deps_type=CustomerDetails,
  retries=3,
  system_prompt=(
    "You are an intelligent customr support agent."
    "Analyze queries carefully and provide structured responses."
    "Always greet the customer and provide a helpful response."
  ),
)


# Add a dynamic system prompt based on dependencies
@agent.system_prompt
async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
  return f'Customer Details: {to_markdown(ctx.deps)}'


customer = CustomerDetails(
  customer_id="1",
  name="Alice Smith",
  email="alice.smith@example.com",
  orders=[
    Order(order_id="12345", status="Shipped", items=["Blue Jeans", "T-Shirt"]),
  ]
)

response = agent.run_sync(user_prompt="What did I order?", deps=customer)
response.all_messages()
print(response.data.model_dump_json(indent=2))

print(
    "Customer Details:\n"
    f"Name: {customer.name}\n"
    f"Email: {customer.email}\n\n"
    "Response Details:\n"
    f"{response.data.response}\n\n"
    "Status:\n"
    f"Follow-up Required: {response.data.follow_up_required}\n"
    f"Needs Escalation: {response.data.needs_escalation}"
)