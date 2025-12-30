import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ============================================
# Load the Fine-tuned Model
# ============================================

model_path = "./student_qa_model"

print("Loading fine-tuned Student Q&A model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Set to evaluation mode
model.eval()
print("Model loaded successfully!\n")

# ============================================
# Context Database for Student Guidance
# ============================================

CONTEXTS = {
    "time_management": """Time management is crucial for academic success. Students should create a daily schedule, prioritize tasks using the Eisenhower matrix, break large assignments into smaller chunks, use techniques like Pomodoro (25 minutes focused work, 5 minutes break), avoid multitasking, set specific study hours, and review their schedule weekly. Consistent sleep schedules and regular breaks improve focus and retention.""",

    "exam_preparation": """Exam preparation requires strategic planning. Start studying at least 2-3 weeks before the exam. Review class notes daily, create summary sheets, use active recall by testing yourself, practice with past papers, form study groups for discussion, teach concepts to others, get adequate sleep before exams (7-8 hours), eat healthy brain foods like nuts and fruits, and stay hydrated. Avoid cramming the night before.""",

    "exam_anxiety": """Dealing with exam anxiety is common among students. Techniques to manage anxiety include deep breathing exercises, progressive muscle relaxation, positive self-talk, adequate preparation, arriving early to the exam venue, reading instructions carefully, starting with easier questions first, managing time during the exam, and accepting that some anxiety is normal. If anxiety is severe, consider speaking with a counselor.""",

    "career_guidance": """Choosing the right career path involves self-assessment, research, and planning. Students should identify their interests using tools like Holland Code assessment, evaluate their skills and strengths, research various career options, talk to professionals in fields of interest, consider job market trends, pursue internships for hands-on experience, align career choice with personal values, and remain open to changing paths as they learn more about themselves.""",

    "note_taking": """Effective note-taking improves learning and retention. Popular methods include the Cornell method (divide page into notes, cues, and summary sections), mind mapping for visual learners, outlining for structured content, the charting method for comparing information, and the sentence method for fast-paced lectures. Review notes within 24 hours, highlight key points, and rewrite notes in your own words for better understanding.""",

    "mental_health": """Maintaining mental health while studying is essential. Students should maintain work-life balance, exercise regularly (at least 30 minutes daily), stay connected with friends and family, pursue hobbies outside academics, practice mindfulness or meditation, seek help when feeling overwhelmed, limit social media usage, set realistic goals, celebrate small achievements, and remember that grades don't define self-worth.""",

    "study_habits": """Building good study habits takes time and consistency. Study at the same time and place daily to build routine, eliminate distractions by turning off phone notifications, use the two-minute rule (if a task takes less than two minutes, do it now), reward yourself after completing study goals, track your progress in a journal, find an accountability partner, and create a dedicated study space with good lighting and minimal clutter.""",

    "academic_pressure": """Handling academic pressure requires a balanced approach. Break large goals into smaller milestones, communicate with teachers about difficulties, don't compare yourself to others, focus on personal improvement, take breaks when needed, learn to say no to excessive commitments, maintain perspective that one exam or grade doesn't determine your future, and develop a growth mindset that views challenges as opportunities to learn.""",

    "concentration": """Improving concentration while studying requires environmental and mental preparation. Choose a quiet study location, use noise-canceling headphones or ambient sounds, keep your phone in another room, use website blockers for distracting sites, take regular short breaks every 45-50 minutes, stay hydrated, maintain proper posture, ensure good lighting, and practice single-tasking instead of multitasking.""",

    "memory": """Memory techniques help retain information longer. Use spaced repetition to review material at increasing intervals, create mnemonics and acronyms, visualize concepts through mental images, associate new information with what you already know, teach others to reinforce your understanding, get adequate sleep as memory consolidation happens during sleep, use flashcards for quick revision, and practice active recall instead of passive re-reading.""",

    "motivation": """Finding motivation to study when you don't feel like it is challenging. Start with just 5 minutes and build momentum, remember your long-term goals and why you're studying, break tasks into smaller achievable chunks, change your study environment, study with motivated friends, reward yourself after completing tasks, eliminate negative self-talk, visualize success, maintain physical health through exercise and nutrition, and establish a morning routine.""",

    "failure": """Dealing with failure and setbacks is part of the learning journey. Accept that failure is temporary not permanent, analyze what went wrong without self-blame, learn lessons from the experience, talk to someone you trust about your feelings, remember past successes, set new realistic goals, take action to improve, maintain self-compassion, understand that many successful people failed multiple times before succeeding, and view setbacks as redirections."""
}

# ============================================
# Question Answering Function
# ============================================

def answer_question(question, context=None):
    """
    Answer a question using the fine-tuned model.
    If no context provided, searches all contexts.
    """
    if context is None:
        # Search through all contexts and find best answer
        best_answer = None
        best_score = -float('inf')

        for ctx_name, ctx_text in CONTEXTS.items():
            answer, score = get_answer(question, ctx_text)
            if score > best_score:
                best_score = score
                best_answer = answer
                best_context = ctx_name

        return best_answer, best_context
    else:
        answer, score = get_answer(question, context)
        return answer, None

def get_answer(question, context):
    """Get answer from a specific context"""
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding="max_length"
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get start and end positions
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely beginning and end of answer
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    # Calculate confidence score
    score = start_scores[0][start_idx].item() + end_scores[0][end_idx].item()

    # Ensure end is after start
    if end_idx < start_idx:
        end_idx = start_idx

    # Decode the answer
    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer, score

# ============================================
# Interactive Chat Interface
# ============================================

def main():
    print("="*60)
    print("       STUDENT GUIDANCE Q&A BOT")
    print("       Fine-tuned DistilBERT Model")
    print("="*60)
    print("\nI can help you with:")
    print("  - Time management")
    print("  - Exam preparation")
    print("  - Dealing with anxiety")
    print("  - Career guidance")
    print("  - Study techniques")
    print("  - Mental health")
    print("  - Motivation")
    print("  - And more!")
    print("\nType 'quit' to exit\n")
    print("-"*60)

    while True:
        question = input("\n📚 Your Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Good luck with your studies! 🎓")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Get answer
        answer, context_used = answer_question(question)

        print(f"\n🤖 Answer: {answer}")
        if context_used:
            print(f"   (Topic: {context_used.replace('_', ' ').title()})")

if __name__ == "__main__":
    main()
