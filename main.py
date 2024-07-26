import json
import torch
import llama_cpp


torch.cuda.empty_cache()


def parseDict(s):
    try:
        return json.loads(s), None
    except:
        return None, True


model_repository = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
model_path = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

llm = llama_cpp.Llama.from_pretrained(
    repo_id = model_repository,
    filename = model_path,
    verbose=False,
)

messages = [
    {
        "role": "system",
        "content": "You are a annoying teen girl.",
    },
]

while True:
    user_input = input("\nUser: \n")
    if not user_input:
        break

    messages.append({
        "role": "user",
        "content": [
            { "type": "text", "content": user_input },
        ],
    })

    err = True
    message = None

    while err:
        outputs = llm.create_chat_completion(
            messages = messages,
            response_format = {
                "type": "json_object",
            },
            temperature = 0.7,
        )
        print(outputs)
        message = outputs["choices"][0]["message"]
        message["content"], err = parseDict(message["content"][:-2])

    messages.append(message)

    output = message["content"]
    if type(output) == dict:
        output = output["content"]

    print("Assistant: \n" + output)

