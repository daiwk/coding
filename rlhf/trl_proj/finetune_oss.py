from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from trl import SFTConfig
from trl import SFTTrainer


peft_model_id = "gpt-oss-20b-multilingual-reasoner"

def load_ds():
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    return dataset

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
    return model, tokenizer

def try_model(dataset, tokenizer, model):
    # dataset结构
    messages = dataset[0]["messages"]
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    print(conversation)
    # 构造msg
    messages = [
        {"role": "user", "content": "¿Cuál es el capital de Australia?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(output_ids)[0]
    print(response)

def train_model(dataset, tokenizer, model):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    training_args = SFTConfig(
        learning_rate=2e-4,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=2048,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir=peft_model_id,
        report_to="trackio",
    #    push_to_hub=True,
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    #trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")

def generate_once(tokenizer, model, SYSTEM_PROMPT, USER_PROMPT):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    return response
    

def infer():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    # Load the original model first
    model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs).cuda()
    # Merge fine-tuned weights with the base model
    peft_model_id = "gpt-oss-20b-multilingual-reasoner"
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.merge_and_unload()

    REASONING_LANGUAGE = "German"
    SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
    USER_PROMPT = "¿Cuál es el capital de Australia?"  # Spanish for "What is the capital of Australia?"

    response = generate_once(tokenizer, model, SYSTEM_PROMPT, USER_PROMPT)
    print(response)

    REASONING_LANGUAGE = "Chinese"  # or Hindi, or any other language...
    SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
    USER_PROMPT = "What is the national symbol of Canada?"

    response = generate_once(tokenizer, model, SYSTEM_PROMPT, USER_PROMPT)
    print(response)


if __name__ == "__main__":
    dataset = load_ds()
    model, tokenizer = load_model()
    try_model(dataset, tokenizer, model)
    train_model(dataset, tokenizer, model)

