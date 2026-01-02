# Mental Health Symptom Classifier - SFT Experiment

Supervised fine-tuning of Llama-3.2-1B using Tinker API for symptom-to-diagnosis classification.

---

## Experiment Goal

Learn SFT fundamentals by training a 1B model to map symptom descriptions to structured mental health assessments.

---

## Setup
```bash
pip install tinker python-dotenv numpy
echo "TINKER_API_KEY=your_key" > .env
```

---

## Data

**30 training examples** covering 15+ mental health conditions

**Format:**
```json
{
  "input": "I've been feeling really sad for weeks. No energy, sleep too much.",
  "output": "Primary concern: Depression\nKey symptoms: Sadness, fatigue, hypersomnia\n..."
}
```

**Tokenization:**
```
Prompt:     "Symptoms: {input}\nAssessment:"     [weight=0]
Completion: " {output}\n\n"                       [weight=1]
```

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-3.2-1B |
| LoRA Rank | 32 |
| Learning Rate | 1e-4 |
| Optimizer | Adam (β1=0.9, β2=0.95) |
| Epochs | 20 |
| Batch Size | 1 (vanilla SGD) |
| Loss Function | Cross-entropy |
| Total Steps | 600 (20 epochs × 30 examples) |

---

## Training
```bash
python train_mental_health.py
```

**Training loop:**
```python
for epoch in range(20):
    for example in training_data:
        forward_backward([example], "cross_entropy")
        optim_step(AdamParams(learning_rate=1e-4))
```

**No batching, no gradient accumulation, no LR scheduling - vanilla baseline.**

---

## Results

**Loss:**
- Initial: 4.0
- Final: 0.25
- Reduction: 94%

**Inference Quality (3 test cases):**
```
Input: "Sad for weeks, no energy, sleep too much"
Output: "Primary concern: Depression (Major Depressive Disorder)
         Key symptoms: Persistent sadness, loss of interest, hypersomnia, fatigue
         Recommendation: Consult mental health professional"
```

✅ Correct diagnosis  
✅ Accurate symptom extraction  
✅ Proper formatting  
✅ Appropriate recommendations  

**All 3 test cases: Perfect match to expected output**

---

## What Worked

- 1B model sufficient for pattern matching
- 30 examples enough with clear structure
- Vanilla SGD converged well
- LoRA rank=32 good balance
- Weight-based selective training effective

---

## Project Structure
```
├── .env                        # API key
├── mental_health_data.json     # 30 examples + config
├── train_mental_health.py      # Training script
└── README.md
```

---

## Inference
```python
import tinker
from tinker import types

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    model_path="tinker://91899da6-4b01-5fe8-b9aa-cadffa483e37:train:0/sampler_weights/mental_health_classifier"
)

# Your inference code here
```

---

## Key Learnings

**SFT Mechanics:**
- SFT = selective next-token prediction with weights
- Weight=0 for prompt (context), weight=1 for completion (training)
- Cross-entropy loss works well for structured outputs

**Tinker API:**
- Handles distributed training infrastructure
- You control: data, loss, hyperparams, training loop
- Tinker controls: GPU management, gradient computation, failures

**1B Model Capabilities:**
- Good: Pattern matching, structured output, classification
- Limited: Complex reasoning, multi-step logic, novel cases

---

## What's Next

**Immediate:**
- [ ] Test on held-out examples (not in training set)
- [ ] Add batching (batch_size=4-8)
- [ ] Expand to 100+ examples

**Advanced:**
- [ ] Learning rate scheduling
- [ ] Try LoRA rank=64
- [ ] Upgrade to 3B model
- [ ] Compare to 8B baseline

---

## Checkpoint
```
tinker://91899da6-4b01-5fe8-b9aa-cadffa483e37:train:0/sampler_weights/mental_health_classifier
```

**Disclaimer:** Educational experiment only. Not for clinical use.

---

## Tech Stack

- Tinker API (distributed training)
- Llama-3.2-1B (base model)
- LoRA (parameter-efficient fine-tuning)
- Python + NumPy

---

**Training time:** ~20 minutes | **Cost:** ~$0.50