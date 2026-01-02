from dotenv import load_dotenv
import tinker
from tinker import types

load_dotenv()

# Create service client and training client
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32
)

# Get tokenizer
tokenizer = training_client.get_tokenizer()

# Create ONE simple training example
example = {
    "input": "banana split",
    "output": "anana-bay plit-say"
}

# Tokenize
prompt = f"English: {example['input']}\nPig Latin:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)

# Create weights
prompt_weights = [0] * len(prompt_tokens)      # Don't train on prompt
completion_weights = [1] * len(completion_tokens)  # DO train on completion

# Combine everything
all_tokens = prompt_tokens + completion_tokens
all_weights = prompt_weights + completion_weights

print(f"Total tokens: {len(all_tokens)}")
print(f"Prompt tokens (weight=0): {len(prompt_weights)}")
print(f"Completion tokens (weight=1): {len(completion_weights)}")

# Shift for next-token prediction
input_tokens = all_tokens[:-1]
target_tokens = all_tokens[1:]
weights = all_weights[1:]

print(f"\nInput tokens: {len(input_tokens)}")
print(f"Target tokens: {len(target_tokens)}")
print(f"Weights: {len(weights)}")

# Create the Datum
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "weights": weights,
        "target_tokens": target_tokens
    }
)

print("\n✅ Datum created successfully!")
print(f"Model input length: {datum.model_input.length}")