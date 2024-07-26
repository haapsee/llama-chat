import json
import torch
import llama_cpp


torch.cuda.empty_cache()

model_repository = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
model_path = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

messages = [
    {
        "role": "system",
        "content": "You are useless personal assistant who knows nothing, is terrible at math but atleast you are funny.",
    },
]

llm = llama_cpp.Llama.from_pretrained(
    repo_id = model_repository,
    filename = model_path,
    verbose = False,
    n_ctx = 2**16,
    n_gpu_layers = 32 if torch.cuda.is_available() else 0,
)


def parseDict(s):
    try:
        return json.loads(s), None
    except:
        return None, True


def generateAssistantReponse():
        message, err = None, True

        while err:
            outputs = llm.create_chat_completion(
                messages = messages,
                response_format = {
                    "type": "json_object",
                },
                temperature = 0.7,
            )
            message = outputs["choices"][0]["message"]
            message["content"], err = parseDict(message["content"])
        return message


def chatLoop():
    while True:
        user_input = input("\nUser: \n")
        if not user_input:
            break

        messages.append({ "role": "user", "content": [{ "type": "text", "content": user_input }]})
        message = generateAssistantReponse()
        messages.append(message)

        output = message["content"]
        if type(output) == dict:
            output = output["content"]

        print("Assistant: \n" + output)


if __name__ == "__main__":
    chatLoop()
