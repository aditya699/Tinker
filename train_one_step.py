from dotenv import load_dotenv
import tinker
from tinker import types
import numpy as np

load_dotenv()

# Setup
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32
)
tokenizer = training_client.get_tokenizer()

# Prepare ONE training example
example = {"input": "banana split", "output": "anana-bay plit-say"}
prompt = f"English: {example['input']}\nPig Latin:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)

# Create weights and datum
prompt_weights = [0] * len(prompt_tokens)
completion_weights = [1] * len(completion_tokens)
all_tokens = prompt_tokens + completion_tokens
all_weights = prompt_weights + completion_weights

input_tokens = all_tokens[:-1]
target_tokens = all_tokens[1:]
weights = all_weights[1:]

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={"weights": weights, "target_tokens": target_tokens}
)

print("Step 1: Training...")
# Do ONE training step
fwd_bwd_future = training_client.forward_backward([datum], "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

fwd_bwd_result = fwd_bwd_future.result()
optim_result = optim_future.result()

# Calculate loss
logprobs = fwd_bwd_result.loss_fn_outputs[0]['logprobs'].tolist()
loss = -np.dot(logprobs, weights) / sum(weights)
print(f"✅ Training completed! Loss: {loss:.4f}\n")

print("Step 2: Saving permanent checkpoint...")
# Save PERMANENT checkpoint
save_result = training_client.save_weights_for_sampler(name="pig_latin_step1").result()
checkpoint_path = save_result.path
print(f"✅ Checkpoint saved!")
print(f"   Path: {checkpoint_path}\n")

print("Step 3: Creating sampling client from checkpoint...")
# Create sampling client from the saved checkpoint
sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
print("✅ Sampling client ready!\n")

print("Step 4: Testing inference...")
# Test on the SAME example
test_prompt1 = "English: banana split\nPig Latin:"
test_tokens1 = tokenizer.encode(test_prompt1, add_special_tokens=True)

result1 = sampling_client.sample(
    prompt=types.ModelInput.from_ints(test_tokens1),
    sampling_params=types.SamplingParams(
        max_tokens=20,
        temperature=0.0,
        stop=["\n"]
    ),
    num_samples=1
).result()

output1 = tokenizer.decode(result1.sequences[0].tokens)
print(f"Test 1 - Trained example:")
print(f"  Input:  'banana split'")
print(f"  Output: '{output1}'")
print(f"  Expected: 'anana-bay plit-say'\n")

# Test on a NEW example
test_prompt2 = "English: coffee break\nPig Latin:"
test_tokens2 = tokenizer.encode(test_prompt2, add_special_tokens=True)

result2 = sampling_client.sample(
    prompt=types.ModelInput.from_ints(test_tokens2),
    sampling_params=types.SamplingParams(
        max_tokens=20,
        temperature=0.0,
        stop=["\n"]
    ),
    num_samples=1
).result()

output2 = tokenizer.decode(result2.sequences[0].tokens)
print(f"Test 2 - New example:")
print(f"  Input:  'coffee break'")
print(f"  Output: '{output2}'")
print(f"  Expected: 'offee-cay eak-bray'\n")

print("=" * 60)
print("To view this checkpoint in the cloud, run:")
print(f"  Set API key: $env:TINKER_API_KEY=\"your_key\"")
print(f"  List checkpoints: tinker checkpoint list")
print("=" * 60)