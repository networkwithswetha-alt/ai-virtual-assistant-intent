# AI Virtual Assistant — Intent Classification

> AI coursework-style portfolio project built from scratch with synthetic data.

## What it is

A miniature virtual assistant ("Ava") that understands what a user *wants*
and responds accordingly — the same intent-classification pattern behind
customer-service chatbots and voice assistants.

## How intent classification works

Every user message is mapped to one of 8 **intents**:

| Intent | Example utterance |
|---|---|
| `greeting` | "hi, I need some help" |
| `book_appointment` | "book a slot for Tuesday at 2pm" |
| `check_status` | "what is the status of my request?" |
| `reset_password` | "I forgot my password" |
| `product_info` | "tell me about your credit card" |
| `complaint` | "I want to file a complaint" |
| `goodbye` | "thanks, bye" |
| `fallback` | "asdfghjkl" (nonsense / out of scope) |

The pipeline: **TF-IDF vectorization → logistic regression**. The classifier
outputs a probability per intent; if the top confidence is below a threshold
(0.35), the assistant asks for clarification instead of guessing wrong.

## Architecture

```
                    ┌─────────────────────┐
  user message ───▶ │  TF-IDF + LogReg    │ ──▶ (intent, confidence)
                    │  intent classifier  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Dialogue manager   │  tracks state:
                    │  (assistant.py)     │  user name, pending
                    └─────────┬───────────┘  appointment day
                              │
                    ┌─────────▼───────────┐
                    │ Response templates  │ ──▶ assistant reply
                    │ per intent          │
                    └─────────────────────┘
```

The **dialogue manager** (`assistant.py`) adds the conversational layer a raw
classifier lacks: it remembers your name, walks multi-turn flows (booking an
appointment takes two turns), and personalizes replies.

## How to run

```bash
pip install -r requirements.txt
python train.py   # train classifier, save confusion matrix to images/
python demo.py    # chat in the terminal
streamlit run app.py   # chat in the browser (shows detected intent)
```

Try this in the demo:
```
You: my name is Priya
You: I want to book an appointment
You: Tuesday
You: what is the status of my transfer issue?
You: blah blah blah
You: quit
```

## Limitations & how I'd extend it

- **Template responses, not generation**: replies come from fixed templates.
  A production system would use an LLM for natural phrasing while keeping
  the classifier for routing.
- **Synthetic training data**: template-generated utterances are cleaner
  than real user messages (typos, code-switching, slang).
- **No entity extraction**: dates/products are matched with simple rules;
  a real NLU stack would add named-entity recognition.
- **Single-turn memory only**: no persistent user profiles or conversation
  history across sessions.

Natural next steps: swap templates for LLM-generated replies, add entity
extraction, and evaluate on real (anonymized) chat logs.

## Project structure

```
ai-virtual-assistant-intent/
├── data.py        # synthetic utterance dataset generator
├── train.py       # classifier training, evaluation, confusion matrix
├── assistant.py   # VirtualAssistant: classify + dialogue state + replies
├── demo.py        # terminal chat demo
├── app.py         # Streamlit chat UI
├── images/        # confusion matrix
├── requirements.txt
└── README.md
```

## Skills demonstrated

Python · NLP · scikit-learn (TF-IDF, text classification) ·
conversational AI concepts (intents, confidence thresholds, dialogue state) ·
Streamlit
