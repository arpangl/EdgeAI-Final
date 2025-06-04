from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. Set model name — change to local path or HuggingFace repo if needed
model_name = "./merged_lora_llama_train_for_3epoch"

# 2. Load the model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 3. Quantize the model using AWQ
quant_config = {
    "zero_point": True,         # enables zero-point quantization
    "q_group_size": 128,        # recommended: 128 or 32
    "w_bit": 4,                 # number of weight bits: typically 4
    "version": "GEMM"           # GEMM backend for fast inference
}
model.quantize(tokenizer, quant_config=quant_config)

# 4. Save the quantized model
output_dir = "./llama3-3b-instruct-awq"
model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Quantized model saved at {output_dir}")
