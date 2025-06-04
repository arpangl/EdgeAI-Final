import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

#LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftMixedModel




#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()



def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    
    
    # --- Configuration ---
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    max_new_tokens = 256  # Number of new tokens to generate (This is for generation, not directly for QLoRA training)
    device = 'cuda:0'
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # --- 1. Load Model with 4-bit Quantization (QLoRA prerequisite) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device, # Use device_map for distributed loading
        torch_dtype=torch.bfloat16, # Recommended for Llama models with bnb_4bit_compute_dtype
    )

    # --- 2. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token for models like Llama that don't have one by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare Model for k-bit Training (QLoRA) ---
    # This reconfigures the model for gradient checkpointing and adds adapters
    model = prepare_model_for_kbit_training(model)

    # --- 4. Configure LoRA ---
    # These are common LoRA parameters. You might need to tune them.
    lora_config = LoraConfig(
        r=32,  # LoRA attention dimension
        lora_alpha=64,  # Alpha parameter for LoRA scaling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head", # Include lm_head for instruction tuning, might be optional for pure language modeling
        ],
        bias="none",  # Don't train bias weights
        lora_dropout=0.05,
        task_type="CAUSAL_LM", # Specify task type
    )

    # --- 5. Get PEFT Model ---
    model = get_peft_model(model, lora_config)

    # Print trainable parameters (should be much smaller now)
    model.print_trainable_parameters()

    # --- 6. Data Preprocessing ---
    def tokenize_function(examples):
        # Ensure truncation is handled correctly, especially for long texts
        return tokenizer(examples["text"], truncation=True, max_length=max_new_tokens) # Adjust max_length as needed

    # Tokenize the training dataset
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4, # Use multiple processes for faster tokenization
        remove_columns=["text"] # Remove the original text column after tokenization
    )

    # --- 7. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./qlora_results",
        num_train_epochs=3, # You might need to adjust this
        per_device_train_batch_size=1, # Adjust based on your GPU memory
        gradient_accumulation_steps=1, # Simulate larger batch size
        optim="paged_adamw_8bit", # Recommended optimizer for QLoRA
        save_strategy="no",
        logging_steps=200,
        learning_rate=1e-5,
        weight_decay=0.001,
        fp16=False, # Set to False since we are using bfloat16 for compute_dtype
        bf16=True, # Enable bfloat16
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
        report_to="tensorboard", # Or "none"
    )

    # --- 8. Train the Model ---
    from trl import SFTTrainer

    # SFTTrainer is often used for fine-tuning with QLoRA
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_train_dataset,
        args=training_args,
    )

    trainer.train()

    # --- 9. Save the PEFT adapters ---
    trainer.save_model("qlora_llama_adapters_3epoch")

    # --- Optional: Merge and Save Full Model (for inference) ---
    # This part is for after training if you want a full model for deployment
    # You would load the base model again and then merge the adapters
    # from peft import PeftModel
    #
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    model_to_save = PeftMixedModel.from_pretrained(base_model, "qlora_llama_adapters_3epoch")
    model_to_save.save_pretrained("./merged_qlora_llama_model_3epoch")
    tokenizer.save_pretrained("./merged_qlora_llama_model_3epoch")













    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_new_tokens + 16,
        device=model.device,
        dtype=torch.float16
    )
    ####################################################################

    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up ===
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()





    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing ===
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)







    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')

    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])

if __name__ == '__main__':
    main()