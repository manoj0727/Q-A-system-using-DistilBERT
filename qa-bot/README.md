# Student Guidance Q&A Bot

A fine-tuned DistilBERT model for answering student-related questions about study techniques, exam preparation, career guidance, mental health, and more.

## Project Structure

```
qa-bot/
├── train.py              # Training script
├── inference.py          # Interactive Q&A interface
├── data/
│   └── student_qa_dataset.json  # Training dataset
├── student_qa_model/     # Saved model (after training)
└── requirements.txt      # Dependencies
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install transformers datasets torch accelerate
```

## Training

Run the training script:
```bash
python train.py
```

This will:
- Load the student Q&A dataset (60+ Q&A pairs)
- Fine-tune DistilBERT for 10 epochs
- Save the model to `./student_qa_model`

## Usage

After training, run the interactive Q&A bot:
```bash
python inference.py
```

Example questions:
- "How can I manage my time better?"
- "How should I prepare for exams?"
- "How do I deal with exam anxiety?"
- "How can I improve my memory?"
- "What should I do after failing an exam?"

## Topics Covered

- Time Management
- Exam Preparation
- Exam Anxiety
- Career Guidance
- Note-taking Methods
- Mental Health
- Study Habits
- Academic Pressure
- Concentration
- Memory Techniques
- Motivation
- Dealing with Failure

## Model Details

- Base Model: `distilbert-base-uncased`
- Parameters: 66 million
- Task: Extractive Question Answering
- Training Epochs: 10
- Learning Rate: 2e-5

## Resume Bullet

> Fine-tuned DistilBERT on 60+ domain-specific Q&A pairs using HuggingFace Transformers for student guidance chatbot
