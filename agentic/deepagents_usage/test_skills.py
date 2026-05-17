from pathlib import Path
from urllib.request import urlopen

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import MemorySaver

PYTHON = "/opt/homebrew/opt/python@3.11/bin/python3.11"

checkpointer = MemorySaver()

root_dir = Path("./agent_workspace").resolve()
skill_dir = root_dir / "skills" / "arxiv-search"
skill_dir.mkdir(parents=True, exist_ok=True)

skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/arxiv-search/SKILL.md"
skill_py_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/arxiv-search/arxiv_search.py"

with urlopen(skill_url) as response:
    skill_content = response.read().decode("utf-8")

with urlopen(skill_py_url) as response:
    skill_py_content = response.read().decode("utf-8")

(skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
(skill_dir / "arxiv_search.py").write_text(skill_py_content, encoding="utf-8")

backend = LocalShellBackend(root_dir=str(root_dir))

print("=== shell debug ===")
print(backend.execute("pwd").output)
print(backend.execute(f"{PYTHON} --version").output)
print(backend.execute("ls -R skills").output)

print("=== install arxiv ===")
print(backend.execute(f"{PYTHON} -m pip install arxiv").output)

agent = create_deep_agent(
    model="ollama:gpt-oss:120b-cloud",
    backend=backend,
    skills=[str(root_dir / "skills")],
    checkpointer=checkpointer,
    system_prompt=(
        "你是一个研究助手。"
         "如果任务涉及 arXiv 论文检索，必须使用已注册的 arxiv-search skill，"
        "当需要执行本地 Python 脚本时，必须使用 "
        "/opt/homebrew/opt/python@3.11/bin/python3.11。"
        "注意：execute 的工作目录是 backend root_dir；"
        "执行本地 skill 脚本时使用相对路径，例如 "
        "skills/arxiv-search/arxiv_search.py，"
        "不要在 execute 命令里使用 /skills/... 绝对路径。"
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "帮我查一下arxiv上关于大模型+推荐的的5篇论文。"
                ),
            }
        ],
    },
    config={"configurable": {"thread_id": "12345"}},
)

print("=== agent result ===")
print(result["messages"][-1].content)