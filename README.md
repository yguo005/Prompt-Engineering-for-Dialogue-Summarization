# Prompt Engineering for Dialogue Summarization

This repository contains a comprehensive assignment on **prompt engineering** techniques for dialogue summarization using the Flan-T5 model and the DialogSum dataset.

##  Overview

This assignment explores how different prompting strategies affect the performance of large language models on dialogue summarization tasks. Students will implement and compare:

- **Zero-shot inference** with instruction prompts
- **Few-shot inference** with in-context examples
- **Different prompt templates** and their impact
- **Generation configuration parameters** for text generation

## Objectives


- Understand the fundamentals of prompt engineering
- Learn the difference between zero-shot and few-shot inference
- Explore how prompt structure affects model outputs
- Gain hands-on experience with Hugging Face Transformers
- Practice working with real-world dialogue datasets

##  File Structure

```
├── PA5_(Prompt_Engineering).ipynb  # Main assignment notebook
└── README.md                                # This file
```

##  Requirements

### Dependencies
```bash
pip install transformers datasets torch
```

### Hardware Requirements
- GPU recommended (T4 or better) for faster inference
- Minimum 8GB RAM
- Google Colab compatible

##  Dataset

The assignment uses the **DialogSum dataset** (`knkarthick/dialogsum`), which contains:
- **12,460** training dialogues
- **500** validation dialogues  
- **1,500** test dialogues
- Human-written summaries for each conversation

### Sample Data Format
```
Dialogue: "#Person1#: Ms. Dawson, I need you to take a dictation for me..."
Summary: "Ms. Dawson helps #Person1# to write a memo about changing communication methods."
```

##  Model

**Flan-T5-Large** (`google/flan-t5-large`)
- 783M parameters
- Instruction-tuned version of T5
- Optimized for following natural language instructions

##  Assignment Structure

### 1. Dataset Exploration
- Load and examine the DialogSum dataset
- Understand dialogue structure and summary patterns
- Analyze different conversation types

### 2. Baseline Model (No Prompt Engineering)
```python
# Direct model inference without specific instructions
summary = model.generate(tokenized_dialogue, max_new_tokens=50)
```

### 3. Zero-Shot Inference with Instruction Prompts
Test various instruction formats:
- `"Summarize the following conversation."`
- `"Please provide a brief summary of this dialogue."`
- `"What is the main point of this conversation?"`

### 4. Flan-T5 Pre-built Templates
Experiment with official Flan-T5 prompt templates:
```python
templates = [
    "{dialogue}\n\nBriefly summarize that dialogue.",
    "Here is a dialogue:\n{dialogue}\n\nWrite a short summary!",
    # ... more templates
]
```

### 5. Few-Shot Inference
Implement in-context learning with multiple examples:
```python
def make_prompt(in_context_examples, test_example):
    # Provide 2-3 example dialogues with summaries
    # Follow with the target dialogue (no summary)
    return formatted_prompt
```

### 6. Generation Configuration Parameters
Explore different decoding strategies:
```python
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=50
)
```



##  Key Techniques Covered

### Prompt Engineering Strategies
- **Instruction clarity**: Clear vs. ambiguous prompts
- **Prompt endings**: Empty string vs. "Summary:" cue
- **Template variations**: Different phrasings for same task
- **Context length**: Balancing examples vs. token limits

### Few-Shot Learning
- **Example selection**: Choosing representative dialogues
- **Example formatting**: Consistent structure across examples
- **Context window management**: Fitting multiple examples in 512 tokens

### Generation Control
- **Temperature**: Controls randomness (0.1 = focused, 1.0 = creative)
- **Top-p sampling**: Nucleus sampling for better coherence
- **Top-k sampling**: Limiting vocabulary choices
- **Repetition penalty**: Reducing redundant phrases

