"""Simple chat script for Qwen3.5-4B-Base."""

import argparse
import re

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "hf_qwen"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3.5-4B-Base")
    parser.add_argument(
        "--no-think", action="store_true", help="Disable thinking mode"
    )
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )

    print("Model loaded. Type 'quit' to exit.\n")

    messages = []
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.strip().lower() in ("quit", "exit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not args.no_think,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print("Generating...", end="", flush=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = strip_thinking(raw_response)

        print(f"\rAssistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
