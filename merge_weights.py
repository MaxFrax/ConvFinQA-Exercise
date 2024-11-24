from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('adapter_path', type=str)
parser.add_argument('output_folder', type=str)
parser.add_argument('--hf_name', type=str)

args = parser.parse_args()

MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = PeftModel.from_pretrained(model, args.adapter_path)

print(model.active_adapter)

model = model.merge_and_unload()
model.save_pretrained(args.output_folder)
if args.hf_name:
    model.push_to_hub(f'maxfrax/{args.hf_name}',private=True, token=HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(f'maxfrax/{args.hf_name}', private=True, token=HF_TOKEN)