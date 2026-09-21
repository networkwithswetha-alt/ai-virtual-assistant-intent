"""Command-line chat demo for the virtual assistant.

Usage:
    python demo.py
Type 'quit' or 'exit' to end the conversation.
"""

from assistant import VirtualAssistant


def main():
    bot = VirtualAssistant()
    print("Ava (virtual assistant) — type 'quit' to exit.\n")
    print("Ava:", bot.respond("hello"))
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAva: Goodbye!")
            break
        if text.lower() in {"quit", "exit"}:
            print("Ava:", bot.respond("goodbye"))
            break
        if not text:
            continue
        print("Ava:", bot.respond(text))


if __name__ == "__main__":
    main()
