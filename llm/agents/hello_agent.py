from typing import List, Dict
from llm.client_manager import ClientManager
from utils.parameters import parse_parameters

class HelloAgentsLLM:

    def __init__(self):
        self.args = parse_parameters()
        self.client = ClientManager(args=self.args)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:

        print(f"🧠 Calling {self.args.model} model...")
        try:

            response = self.client.chat_completion(
                messages=messages,
            )

            return response
        except Exception as e:
            print(f"❌ Error while calling LLM API: {e}")
            return None

if __name__ == '__main__':

    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a quicksort algorithm"}
        ]

        print("--- Calling LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- Full model response ---")
            print(responseText)

    except ValueError as e:
        print(e)
