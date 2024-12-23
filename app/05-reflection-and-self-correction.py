from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.gemini import GeminiModel
# from pydantic_ai.models.openai import OpenAIModel
from typing import Dict, List, Optional

# Fix the utils.markdown import error
import sys
sys.path.append("app")

from utils.markdown import to_markdown

model = GeminiModel("gemini-1.5-flash")


"""
This example demonstrates advanced agent capabilities such as reflection and self-correction.

key concepts:
- Implementing self-reflection in agents
- Handling errors gracefully with retries
- Using ModelRetry for automatic retries
- Implementing self-correction in agents
- Decorator-based tool integration
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


shipping_info_db: Dict[str, str] = {
  "#12345": "Shipped on 2024-12-01",
  "#67890": "Out for delivery",
  "#54321": "Processing",
}

customer = CustomerDetails(
  customer_id="1",
  name="Alice Smith",
  email="alice.smith@example.com",
)


agent = Agent(
  model=model,
  result_type=ResponseModel,
  deps_type=CustomerDetails,
  retries=3,
  system_prompt=(
    "You are an intelligent customr support agent."
    "Analyze queries carefully and provide structured responses."
    "Use tools to look up relevant information."
    "Always greet the customer and provide a helpful response."
  ),
)


@agent.tool_plain() # Add plain tool via decorator
def get_shipping_status(order_id: str) -> str:
  """Get the shipping status for a given order ID."""
  shipping_status = shipping_info_db.get(order_id)
  if shipping_status is None:
    raise ModelRetry(
      f"No shipping information found for order ID: {order_id}"
      "Make sure the order ID starts with a #. e.g, #67890"
      "Self-correct this if needed and try again."
    )
  return shipping_info_db[order_id]


response = agent.run_sync(
  user_prompt="What's the status of my last order 12345?",
  deps=customer
)

response.all_messages()
print(response.data.model_dump_json(indent=2))
