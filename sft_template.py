from dotenv import load_dotenv
import tinker
from tinker import types
import numpy as np
import json

load_dotenv()

print("="*60)
print("MENTAL HEALTH SYMPTOM CLASSIFIER - SFT TRAINING")
print("="*60)

# Load configuration
print("\n[1/6] Loading configuration...")
with open("mental_health_data.json", "r") as f:
    config = json.load(f)

print(f"✅ Loaded configuration:")
print(f"   Model: {config['model']}")
print(f"   LoRA Rank: {config['lora_rank']}")
print(f"   Learning Rate: {config['learning_rate']}")
print(f"   Training Examples: {len(config['examples'])}")
print(f"   Epochs: {config['num_epochs']}")

# Setup Tinker
print("\n[2/6] Connecting to Tinker...")
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=config['model'],
    rank=config['lora_rank']
)
tokenizer = training_client.get_tokenizer()
print("✅ Connected to Tinker!")

# Prepare training data
print("\n[3/6] Preparing training data...")

def create_datum(example):
    """Convert a training example into a Datum object"""
    # Format: "Symptoms: <input>\nAssessment:"
    prompt = f"Symptoms: {example['input']}\nAssessment:"
    
    # Tokenize prompt and completion
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    
    # Create weights (0 for prompt, 1 for completion)
    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)
    
    # Combine
    all_tokens = prompt_tokens + completion_tokens
    all_weights = prompt_weights + completion_weights
    
    # Shift for next-token prediction
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"weights": weights, "target_tokens": target_tokens}
    )

training_data = [create_datum(ex) for ex in config['examples']]
print(f"✅ Prepared {len(training_data)} training examples")

# Show example of what the data looks like
print("\n[Example of first training case]")
first_ex = config['examples'][0]
print(f"Input: {first_ex['input'][:80]}...")
print(f"Output: {first_ex['output'][:80]}...")

# Training loop
print(f"\n[4/6] Starting training for {config['num_epochs']} epochs...")
print("="*60)

total_steps = len(training_data) * config['num_epochs']
step = 0
all_losses = []

for epoch in range(config['num_epochs']):
    epoch_losses = []
    
    for i, datum in enumerate(training_data):
        step += 1
        
        # Training step (vanilla - one example at a time)
        fwd_bwd_future = training_client.forward_backward([datum], "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=config['learning_rate']))
        
        # Wait for results
        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()
        
        # Calculate loss - FIXED: Convert TensorData to list
        logprobs = fwd_bwd_result.loss_fn_outputs[0]['logprobs'].tolist()
        weights_list = datum.loss_fn_inputs['weights'].tolist()  # Convert TensorData to list
        loss = -np.dot(logprobs, weights_list) / sum(weights_list)
        epoch_losses.append(loss)
        all_losses.append(loss)
        
        # Progress update every 10 steps
        if step % 10 == 0 or step == total_steps:
            recent_avg = np.mean(all_losses[-10:])
            print(f"Step {step:3d}/{total_steps} | Epoch {epoch+1:2d}/{config['num_epochs']} | Recent Loss: {recent_avg:.4f}")
    
    # Epoch summary
    avg_epoch_loss = np.mean(epoch_losses)
    print(f">>> Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")
    print("-"*60)

print("="*60)
print("✅ Training completed!")
print(f"Final average loss: {np.mean(all_losses[-30:]):.4f}")

# Save checkpoint
print(f"\n[5/6] Saving model checkpoint...")
save_result = training_client.save_weights_for_sampler(name=config['checkpoint_name']).result()
checkpoint_path = save_result.path
print(f"✅ Model saved!")
print(f"   Path: {checkpoint_path}")

# Test inference
print(f"\n[6/6] Testing model on sample cases...")
print("="*60)

sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)

# Test on 3 examples from training data
test_cases = [
    config['examples'][0],  # Depression
    config['examples'][1],  # Anxiety
    config['examples'][4],  # PTSD
]

for i, test_case in enumerate(test_cases, 1):
    test_prompt = f"Symptoms: {test_case['input']}\nAssessment:"
    test_tokens = tokenizer.encode(test_prompt, add_special_tokens=True)
    
    result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(test_tokens),
        sampling_params=types.SamplingParams(
            max_tokens=150,
            temperature=0.0,  # Greedy sampling for consistency
            stop=["\n\n"]
        ),
        num_samples=1
    ).result()
    
    output = tokenizer.decode(result.sequences[0].tokens)
    
    print(f"\nTest Case {i}:")
    print(f"Input: {test_case['input'][:60]}...")
    print(f"Model Output:\n{output}")
    print("-"*60)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nYour model is ready to use:")
print(f"Checkpoint: {checkpoint_path}")
print(f"\nTo view in cloud: tinker checkpoint list")
print("="*60)