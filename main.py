import json
import torch
import llama_cpp


torch.cuda.empty_cache()

prompt = """
You are a helpful assistant with tool calling capabilities.

If you are using tools, respond in the format {"name": function name, "parameters": dictionary of function arguments}. Do not use variables.

You have access to the following functions:

Use the function 'get_current_weather' to get the current weather conditions for a specific location

{
    "type": "function",
    "function": {
    "name": "get_current_weather",
    "description": "Get the current weather conditions for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["Celsius", "Fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location."
            }
            },
            "required": ["location", "unit"]
        }
    }
}

Use the function 'get_current_traffic' to get the current traffic conditions for a specific location
{
    "type": "function",
    "function": {
    "name": "get_current_traffic",
    "description": "Get the current traffic conditions for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA"
            },
            "required": ["location"]
        }
    }
}

"""

model_repository = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
model_path = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

messages = [{ "role": "system", "content": prompt }]

llm = llama_cpp.Llama.from_pretrained(
    repo_id = model_repository,
    filename = model_path,
    verbose = False,
    n_ctx = 2**12,
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

        output = message.get("content")

        if output.get("name"):
            output = "Query: " + json.dumps(output)

        if type(output) == dict:
            output = output["content"]

        print("Assistant: \n" + output)


if __name__ == "__main__":
    chatLoop()
