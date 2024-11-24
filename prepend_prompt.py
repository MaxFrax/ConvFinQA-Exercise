import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Generate text from a model')
parser.add_argument('prompt_file', type=str, help='Path to the prompt file')
parser.add_argument('dataset', type=str, help='Path to the dataset')

args = parser.parse_args()

prompt = open(args.prompt_file, 'r').read()

dataset = pd.read_parquet(args.dataset)


system_prompt = {
                    'content': prompt,
                    'role': 'system'
                }

dataset['messages'] = dataset['messages'].apply(lambda x: [system_prompt] + eval(x))

dir_name = os.path.dirname(args.dataset)
file_name = os.path.basename(args.dataset)
new_path = os.path.join(dir_name, 'prompted_' + file_name)
dataset.to_parquet(new_path, index=False)