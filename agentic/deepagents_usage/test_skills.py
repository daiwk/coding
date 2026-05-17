from pathlib import Path
from urllib.request import urlopen

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import MemorySaver


# PYTHON = "~/envtest/bin/python3.11"
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
    skills=["/skills/"],
    checkpointer=checkpointer,
    system_prompt=(
        "你是一个可以使用本地 shell 的研究助手。"
        "当需要使用 arxiv-search skill 时，先 read_file 查看 "
        "/skills/arxiv-search/SKILL.md，"
        "然后用 execute 运行 /skills/arxiv-search/arxiv_search.py。"
        "注意：execute 的工作目录是 backend root_dir。"
        "运行 Python 时必须使用 /opt/homebrew/opt/python@3.11/bin/python3.11，禁止使用 python 或 python3。"
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "帮我查一下2026年5月arxiv上最新的5篇大模型论文。"
                    "请使用 /skills/arxiv-search 里的 skill 脚本。"
                    "运行脚本时必须使用 /opt/homebrew/opt/python@3.11/bin/python3.11。"
                    "建议查询："
                    'all:"large language models" AND submittedDate:[202605010000 TO 202605312359]，'
                    "max-papers 设为 5。"
                ),
            }
        ],
    },
    config={"configurable": {"thread_id": "12345"}},
)

print("=== agent result ===")
print(result["messages"][-1].content)