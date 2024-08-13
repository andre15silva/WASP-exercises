from datasets import load_dataset

dataset = load_dataset("ASSERT-KTH/megadiff-sf-synthetic_test_error", split="train")
dataset = dataset.train_test_split(test_size=0.02)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token

def format_texts(examples, begin_inst="[INST]", end_inst="[\INST]"):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = f"<s>{begin_inst} {examples['prompt'][i]} {end_inst} {examples['answer'][i]}</s>"
        output_texts.append(text)
    return output_texts

def tokenize_function(batch):
    return tokenizer(format_texts(batch), padding=False, truncation=False)

max_seq_length = 4096

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) < max_seq_length, batched=False)

print(f"Training dataset size: {len(tokenized_dataset['train'])}")
print(f"Validation dataset size: {len(tokenized_dataset['test'])}")

from trl import DataCollatorForCompletionOnlyLM

response_template_with_context = "[\INST]"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)

collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from accelerate import PartialState

import torch

device_string = PartialState().process_index
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf",
                                             torch_dtype=torch.bfloat16,
                                             device_map={'':device_string})

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir='tmp_trainer',
    learning_rate=5e-4,
    num_train_epochs=1,
    max_seq_length=max_seq_length,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    packing=False,
    bf16=True,
)

trainer = SFTTrainer(
    model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=sft_config,
    peft_config=peft_config,
    data_collator=collator,
)

trainer.train()
trainer.save_state()
trainer.save_model(output_dir="codellama-instruct-repair")
tokenizer.save_pretrained(save_directory="codellama-instruct-repair")
