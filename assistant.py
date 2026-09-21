"""Rule-based dialogue manager on top of the intent classifier.

The VirtualAssistant class:
  - classifies each user message into an intent (with a confidence score)
  - tracks minimal conversation state (user name, pending appointment)
  - responds from hand-written response templates per intent

Low-confidence predictions fall back to a clarification response instead of
guessing — a standard pattern in production assistants.
"""

import os
import re

import joblib

from train import MODEL_PATH, build_pipeline
from data import generate_dataset

CONFIDENCE_THRESHOLD = 0.35

RESPONSES = {
    "greeting": [
        "Hello! I'm Ava, your virtual assistant. How can I help you today?",
    ],
    "book_appointment": [
        "Sure — I can book that for you. Which day works best?",
        "Got it. What day and time would you like the appointment?",
    ],
    "check_status": [
        "Let me check that for you. Could you share your request or ticket ID?",
    ],
    "reset_password": [
        "No problem. I've sent a password reset link to your registered email. "
        "It expires in 30 minutes.",
    ],
    "product_info": [
        "Happy to explain. Which product would you like to know about — "
        "savings account, credit card, home loan, or mobile app?",
    ],
    "complaint": [
        "I'm sorry to hear that. I've logged your complaint and a specialist "
        "will reach out within 24 hours. Your reference ID is CMP-4821.",
    ],
    "goodbye": [
        "You're welcome! Goodbye, and have a great day.",
    ],
    "fallback": [
        "I'm not sure I understood that. I can help with appointments, "
        "request status, password resets, product info, or complaints.",
    ],
}


def _extract_name(text: str):
    m = re.search(r"\bmy name is (\w+)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def _extract_day(text: str):
    days = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]
    for d in days:
        if d in text.lower():
            return d.capitalize()
    return None


def load_classifier():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["pipeline"]
    # Fallback: train a quick in-memory model
    df = generate_dataset(n_per_intent=40, seed=11)
    pipe = build_pipeline()
    pipe.fit(df["utterance"], df["intent"])
    return pipe


class VirtualAssistant:
    """A tiny stateful virtual assistant."""

    def __init__(self):
        self.clf = load_classifier()
        self.state = {"user_name": None, "appointment_day": None,
                      "turns": 0}

    def classify(self, text: str):
        """Return (intent, confidence) for a user message."""
        proba = self.clf.predict_proba([text])[0]
        idx = proba.argmax()
        return self.clf.classes_[idx], float(proba[idx])

    def respond(self, text: str) -> str:
        """Update state from the message and return the assistant reply."""
        self.state["turns"] += 1

        name = _extract_name(text)
        if name:
            self.state["user_name"] = name
            return f"Nice to meet you, {name}! How can I help you today?"

        day = _extract_day(text)
        if day and self.state.get("awaiting_day"):
            self.state["appointment_day"] = day
            self.state["awaiting_day"] = False
            who = self.state["user_name"] or "there"
            return (f"Done! I've booked your appointment for {day}, {who}. "
                    f"You'll get a confirmation shortly.")

        intent, conf = self.classify(text)
        if conf < CONFIDENCE_THRESHOLD:
            intent = "fallback"

        if intent == "book_appointment" and not self.state["appointment_day"]:
            self.state["awaiting_day"] = True

        reply = RESPONSES[intent][0]
        if self.state["user_name"] and intent == "greeting":
            reply = (f"Hello again, {self.state['user_name']}! "
                     f"How can I help you today?")
        return reply
