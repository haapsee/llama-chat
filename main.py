import transformers
import torch


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
)

messages = [
    {
        "role": "system",
        "content": "You are an all knowing wizard who speaks in riddles",
    },
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

while True:
    user_input = input("\nUser: \n")
    if not user_input:
        break

    messages.append({
        "role": "user",
        "content": user_input,
    })
    outputs = pipe(
        messages,
        max_new_tokens=16000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    messages.append({"role": "assistant", "content": assistant_response})
    print("\nAssistant: \n" + assistant_response)
