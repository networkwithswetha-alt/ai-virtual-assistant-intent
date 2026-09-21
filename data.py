"""Synthetic intent-classification dataset for the virtual assistant.

Each intent has template utterances with slot variations (names, dates,
products) filled in randomly. All text is invented in code — no real user
conversations are used.
"""

import random

import pandas as pd

# Slot values used to vary the templates
NAMES = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Anaya"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["10am", "2pm", "4pm", "11:30am", "3pm"]
PRODUCTS = ["savings account", "credit card", "home loan", "mobile app",
            "debit card"]
ISSUES = ["login", "payment", "statement", "transfer", "balance"]

TEMPLATES = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
        "hi, I need some help", "hello, are you there?",
        "hey, can you help me?", "good afternoon",
    ],
    "book_appointment": [
        "I want to book an appointment",
        "Can I schedule a meeting for {day}?",
        "Book a slot for {day} at {time}",
        "I need an appointment with an advisor",
        "Schedule a call with support on {day}",
        "Can we set up a meeting {day} {time}?",
        "I'd like to book time with someone",
        "Reserve a slot for me on {day}",
    ],
    "check_status": [
        "What is the status of my request?",
        "Check the status of my {issue} issue",
        "Has my complaint been resolved?",
        "Where is my application status?",
        "Can you track my service request?",
        "Is my {issue} problem fixed yet?",
        "Give me an update on my case",
        "What's happening with my ticket?",
    ],
    "reset_password": [
        "I forgot my password",
        "Help me reset my password",
        "I can't log in, need a password reset",
        "Reset my account password please",
        "How do I change my password?",
        "My password isn't working",
        "Send me a password reset link",
        "I need to recover my account",
    ],
    "product_info": [
        "Tell me about your {product}",
        "What are the features of the {product}?",
        "How does the {product} work?",
        "I want details on the {product}",
        "What is the interest rate on the {product}?",
        "Compare your {product} options",
        "Do you offer a {product}?",
        "Explain the {product} to me",
    ],
    "complaint": [
        "I want to file a complaint",
        "This service is terrible",
        "I'm very unhappy with my {issue}",
        "I need to speak to a manager",
        "Your {product} charged me incorrectly",
        "Nobody resolved my {issue} problem",
        "This is unacceptable service",
        "I demand a refund for the {product}",
    ],
    "goodbye": [
        "bye", "goodbye", "thanks, bye", "that's all, thanks",
        "see you later", "thanks for your help, bye",
        "done, thank you", "ok bye",
    ],
    "fallback": [
        "asdfghjkl", "what is the weather on mars",
        "tell me a joke about elephants",
        "who won the cricket match in 1857",
        "blah blah blah", "xyz123 !@#",
        "do you like pizza toppings",
        "sing me a song",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        day=rng.choice(DAYS), time=rng.choice(TIMES),
        product=rng.choice(PRODUCTS), issue=rng.choice(ISSUES),
        name=rng.choice(NAMES),
    )


def generate_dataset(n_per_intent: int = 60, seed: int = 11) -> pd.DataFrame:
    """Generate a balanced synthetic utterance dataset."""
    rng = random.Random(seed)
    rows = []
    for intent, templates in TEMPLATES.items():
        for _ in range(n_per_intent):
            rows.append({"utterance": _fill(rng.choice(templates), rng),
                         "intent": intent})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    print(df["intent"].value_counts())
    print("\nExample:\n", df.iloc[0].to_dict())
