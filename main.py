import transformers
import torch

some_tensor = torch.randn(1000, 1000).cuda()
del some_tensor
print("here")

torch.cuda.empty_cache()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipe = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    device_map='auto',
)
#pipe = transformers.pipeline(
#    "text-generation",
#    model=model_id,
#    gpu_layers=0,
#    device_map='auto',
#    load_in_8bit_fp32_cpu_offload=True,
#    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
#)

messages = [
    {
        "role": "system",
        "content": "You are an all knowing wizard who speaks in riddles",
    },
]

#terminators = [
#    pipe.tokenizer.eos_token_id,
#    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#]

while True:
    user_input = input("\nUser: \n")
    if not user_input:
        break

    messages.append({
        "role": "user",
        "content": user_input,
    })
    outputs = pipe(
        tokenizer.encode(messages),
#        max_new_tokens=16000,
#        eos_token_id=terminators,
#        do_sample=True,
#        temperature=0.6,
#        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    messages.append({"role": "assistant", "content": assistant_response})
    print("\nAssistant: \n" + assistant_response)
