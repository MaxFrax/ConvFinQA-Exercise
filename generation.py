from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
from dotenv import load_dotenv
import argparse
from openai import AsyncOpenAI
import os
import asyncio

load_dotenv()

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_ENDPOINT = os.getenv('RUNPOD_ENDPOINT')
MODEL_NAME = os.getenv('MODEL_NAME')

client = AsyncOpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=RUNPOD_ENDPOINT,
    )

async def runpod_model(conversation):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.2,
                top_p=0.1
            )


            generated_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

        return generated_text

async def main():
    parser = argparse.ArgumentParser(description='Generate text from a model')
    parser.add_argument('prompt_file', type=str, help='Path to the prompt file')

    args = parser.parse_args()

    prompt = open(args.prompt_file, 'r').read()

    dataset = pd.read_parquet('data/processed/dev.parquet')


    system_prompt = {
                        'content': prompt,
                        'role': 'system'
                    }

    dataset['reference'] = dataset['messages'].apply(lambda x: eval(x)[-1]['content'])
    dataset['messages'] = dataset['messages'].apply(lambda x: [system_prompt] + eval(x)[:-1])

    generations = await asyncio.gather(*[runpod_model(conversation) for conversation in dataset['messages']])

    dataset['generated_text'] = generations

    dataset = dataset.astype(str)

    output_file = f"generated_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    dataset.to_parquet(output_file, index=False)
    print(f"Generated text saved to {output_file}")

asyncio.run(main())