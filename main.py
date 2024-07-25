import ctransformers
import transformers
import torch


torch.cuda.empty_cache()

model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id_gguf = "TheBloke/Llama-2-7B-Chat-GGUF"
model_file_gguf = "llama-2-7b-chat.Q4_K_M.gguf"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipe = ctransformers.AutoModelForCausalLM.from_pretrained(
    model_id_gguf,
    model_file=model_file_gguf,
    model_type="llama",
    gpu_layers=0,
)

messages = [
    {
        "role": "system",
        "content": "You are an all knowing wizard who speaks in riddles",
    },
]

while True:
    user_input = input("\nUser: \n")
    if not user_input:
        break

    messages.append({
        "role": "user",
        "content": user_input,
    })
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    outputs = pipe(
        inputs,
    )
    messages.append({"role": "assistant", "content": outputs})
    print("\nAssistant: \n" + outputs)
