import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def generate_lora(
    data_file,
    base_model="gpt2",
    output_dir="./lora",
    epochs=4,
    batch_size=2,
    max_length=128,
    lora_r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"]
):
    def load_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                text = obj.get('prompt', '')
                inp = obj.get('input', '')
                ans = obj.get('answer', '')
                full_text = text + ('\n' + inp if inp else '') + ('\n' + ans if ans else '')
                data.append({'text': full_text})
        return data

    raw_data = load_jsonl(data_file)
    dataset = Dataset.from_list(raw_data)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    def tokenize_fn(example):
        enc = tokenizer(
            example['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        labels = [tok_id if m == 1 else -100 for tok_id, m in zip(input_ids, attention_mask)]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=['text'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=5,
        save_strategy='no',
        optim='adamw_torch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")
    
    return output_dir