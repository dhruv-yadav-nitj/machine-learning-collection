import os
import asyncio
from huggingface_hub import InferenceClient, AsyncInferenceClient


async_client = AsyncInferenceClient(
    provider='together',
    api_key=os.getenv('HUGGING_FACE_API_KEY')
)

client = InferenceClient(
    provider='together',
    api_key=os.getenv('HUGGING_FACE_API_KEY')
)

reference_models = [
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1-0528',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Llama-3.2-3B-Instruct'
]

aggregator_model = 'deepseek-ai/DeepSeek-R1-0528'

user_prompt = 'What are 3 fun things to do in SF?'

aggregator_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
"""


async def run_llm(model):
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return None


async def synthesize() -> None:
    responses = await asyncio.gather(*[run_llm(model) for model in reference_models])  # async.gather -> returns a list
    responses = [r for r in responses if r is not None]

    if not responses:
        print('All model calls failed!')
        return None

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {'role': 'system', 'content': aggregator_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(responses)])},
            {'role': 'user', 'content': user_prompt}
        ],
        stream=True
    )
    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(synthesize())
