import os
from together import Together, AsyncTogether
from dotenv import load_dotenv
import asyncio
from typing import Any, List
import json


load_dotenv()

client = Together(
    api_key=os.getenv('TOGETHER_API_KEY')
)

async_client = AsyncTogether(
    api_key=os.getenv('TOGETHER_API_KEY')
)

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


def getFinalSystemPrompt(system_prompt, results: Any) -> str:
    return (
        system_prompt + "\n" + "\n".join([f'{i+1}. {str(item)}' for i, item in enumerate(results)])
    )


async def get_response(model, prompt, prev_responses: Any = None):
    for sleep_time in [1, 2, 4]:
        try:
            messages = ([
                {'role': 'system', 'content': getFinalSystemPrompt(aggregator_prompt, prev_responses)},
                {'role': 'user', 'content': prompt}
            ]
                if prev_responses else [{'role': 'user', 'content': prompt}])

            response = await async_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=512
            )
            return response
        except Exception as e:
            print(f'Model: {model} ; Error: {e}')
            await asyncio.sleep(sleep_time)
    return None


async def run_moa(prompt) -> str:
    results = await asyncio.gather(*[get_response(model, prompt) for model in reference_models])

    results = list(filter(None, results))  # keep only not None responses

    for i in range(1, layers-1):
        results = await asyncio.gather(*[get_response(model, prompt, results) for model in reference_models])
        results = list(filter(None, results))

    final_response = await get_response(aggregator_model, prompt, results)

    return final_response.choices[0].message.content if final_response else str(None)


base_model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'


async def run_base_model(prompt) -> str:
    response = await get_response(base_model, prompt)
    return response.choices[0].message.content if response else str(None)


async def main():
    moa_out = open('alpaca-eval/moa_preds.jsonl', 'w')
    base_out = open('alpaca-eval/baseline_preds.jsonl', 'w')

    with open('alpaca-eval/prompts.jsonl', 'r') as f:
        prompts = [json.loads(line)['instruction'] for line in f.readlines()]

    for i, prompt in enumerate(prompts):
        print(f'\n Processing Prompt {i+1}/{len(prompts)}')

        moa_response, base_response = await asyncio.gather(run_moa(prompt), run_base_model(prompt))

        moa_out.write(json.dumps({'instruction': prompt, 'output': moa_response}) + '\n')
        base_out.write(json.dumps({'instruction': prompt, 'output': base_response}) + '\n')

    moa_out.close()
    base_out.close()


if __name__ == '__main__':
    asyncio.run(main())
