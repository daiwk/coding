import os
from typing import Literal

from deepagents import create_deep_agent
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    #"model": "openai:gpt-5.4",  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model="ollama:qwen3:1.7b",
    subagents=subagents,
)

result = agent.invoke({"messages": [{"role": "user", "content": "帮我查一下2026年5月arxiv上最新的5篇推荐系统论文"}]})

# Print the agent's response
print(result)
print("xxx")
print(result["messages"][-1].content)
