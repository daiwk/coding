#encoding=utf8
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    #"Qwen/Qwen3-0.6B",
    "./llm_model",
    trust_remote_code=True
)

text = "今天天气很好，但我不想上班"

enc = tokenizer(text, add_special_tokens=False) # 保证模型不自己加上eos,bos之类的token
token_texts = []
for tid in enc["input_ids"]:
    token_texts.append(tokenizer.decode([tid]))
print("token num: ", len(enc["input_ids"]))
print("token ids: ", enc["input_ids"])
print("token texts: ", token_texts)
