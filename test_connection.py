from dotenv import load_dotenv
import tinker

# Load environment variables from .env file
load_dotenv()

# Create service client
service_client = tinker.ServiceClient()

# Check available models
print("Available models:")
capabilities = service_client.get_server_capabilities()
for item in capabilities.supported_models:
    print(f"- {item.model_name}")

print("\n✅ Connection successful!")

# Create a training client for a small model
base_model = "meta-llama/Llama-3.2-1B"  # Starting with smallest model
print(f"Creating training client for {base_model}...")

training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32  # LoRA rank
)

print("✅ Training client created successfully!")
print(f"Model: {base_model}")
print(f"LoRA rank: 32")