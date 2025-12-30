import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import Dataset as HFDataset

# ============================================
# STEP 1: Load Custom Student Q&A Dataset
# ============================================

def load_student_qa_data(file_path):
    """Load and process the student Q&A dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_data = []

    for item in data['data']:
        context = item['context']
        for qa in item['qas']:
            question = qa['question']
            answer_text = qa['answer']

            # Find answer start position in context
            answer_start = context.find(answer_text)
            if answer_start == -1:
                # Try case-insensitive search
                answer_start = context.lower().find(answer_text.lower())

            if answer_start != -1:
                processed_data.append({
                    'context': context,
                    'question': question,
                    'answers': {
                        'text': [answer_text],
                        'answer_start': [answer_start]
                    }
                })

    return processed_data

# ============================================
# STEP 2: Initialize Tokenizer and Model
# ============================================

print("Loading DistilBERT tokenizer and model...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print(f"Model loaded: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")

# ============================================
# STEP 3: Preprocess Function
# ============================================

def preprocess_function(examples):
    """Tokenize and prepare data for training"""
    questions = examples["question"]
    contexts = examples["context"]

    # Tokenize
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    # Find start and end positions
    start_positions = []
    end_positions = []

    offset_mapping = tokenized["offset_mapping"]

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        answer_text = answer["text"][0]
        end_char = start_char + len(answer_text)

        # Find token positions
        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            if start is None or end is None:
                continue
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    # Remove offset_mapping
    del tokenized["offset_mapping"]

    return tokenized

# ============================================
# STEP 4: Load and Process Data
# ============================================

print("\nLoading student Q&A dataset...")
qa_data = load_student_qa_data("data/student_qa_dataset.json")
print(f"Loaded {len(qa_data)} Q&A pairs")

# Convert to HuggingFace Dataset
dataset = HFDataset.from_list(qa_data)

# Split into train and validation
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['test'])}")

# Tokenize the dataset
print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("Tokenization complete!")

# ============================================
# STEP 5: Training Configuration
# ============================================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    logging_steps=5,
    warmup_steps=100,
    save_total_limit=2,
    report_to="none"
)

# ============================================
# STEP 6: Initialize Trainer
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# ============================================
# STEP 7: Train the Model
# ============================================

print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

trainer.train()

# ============================================
# STEP 8: Save the Fine-tuned Model
# ============================================

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

model_save_path = "./student_qa_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"\nModel saved to: {model_save_path}")
print("\nTraining complete! Your Student Q&A Bot is ready!")
