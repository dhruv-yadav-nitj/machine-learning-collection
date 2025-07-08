import os
import asyncio
from typing import Optional, List
from together import Together, AsyncTogether, Client, AsyncClient
from dotenv import load_dotenv

load_dotenv()

client = Together(
    api_key=os.getenv('TOGETHER_API_KEY')
)

async_client = AsyncTogether(
    api_key=os.getenv('TOGETHER_API_KEY')
)

user_prompt = input()

reference_models = [
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'Qwen/Qwen2.5-Coder-32B-Instruct',
    'deepseek-ai/DeepSeek-V3'
]

aggregator_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
"""

aggregator_model = 'deepseek-ai/DeepSeek-V3'

layers = 3


def getFinalSystemPrompt(system_prompt, results):
    return (
        system_prompt + "\n" + "\n".join([f'{i+1}. {str(item)}' for i, item in enumerate(results)])
    )


async def run_llm(model, prev_response: Optional[List] = None) -> Optional[str]:
    response = None
    for sleep_time in [1, 2, 4]:
        try:
            if prev_response:
                messages = [
                    {'role': 'system', 'content': getFinalSystemPrompt(aggregator_prompt, prev_response)},
                    {'role': 'user', 'content': user_prompt}
                ]
            else:
                messages = [
                    {'role': 'user', 'content': user_prompt}
                ]
            response = await async_client.chat.completions.create(
                model=model, messages=messages, temperature=0.7, max_tokens=512
            )
            break
        except Exception as e:
            print(f'Model: {model}', e)
            await asyncio.sleep(sleep_time)
    if response:
        return response.choices[0].message.content
    else:
        return None


async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])  # 1st layer

    for i in range(1, layers-1):  # 2nd to (n-1)th layer
        results = await asyncio.gather(*[run_llm(model, results) for model in reference_models])

    results = list(filter(None, results))  # remove all the None responses

    # nth layer
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {'role': 'system', 'content': getFinalSystemPrompt(aggregator_prompt, results)},
            {'role': 'user', 'content': user_prompt}
        ],
        stream=True
    )

    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
