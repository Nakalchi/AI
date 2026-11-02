from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading the DialoGPT model...")

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None
step = 0

print("Chatbot is ready! Type 'exit' to end the chat.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    chat_history_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if step > 0
        else new_input_ids
    )

    response_ids = model.generate(
        chat_history_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        response_ids[:, chat_history_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print(f"Chatbot: {response}")

    step += 1
