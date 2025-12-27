from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    questions = examples["question"]
    contexts = examples["context"]

    # Tokenize questions and contexts
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    # Find start and end token positions for answers
    start_positions = []
    end_positions = []

    offset_mapping = tokenized["offset_mapping"]

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Find token positions
        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    # Remove offset_mapping (not needed for training)
    del tokenized["offset_mapping"]

    return tokenized

# Apply to dataset
print("Processing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("Done!")
print(tokenized_dataset)
