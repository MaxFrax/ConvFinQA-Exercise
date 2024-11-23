import os
import json
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Convert ConvFinQA json to parquet. Does not work on *_turn.json')
parser.add_argument('input', type=str, help='Path to ConvFinQA json file')
args = parser.parse_args()

with open(args.input) as json_file:
        train = json.load(json_file)

dataset = []
file_names = []

# Creates one conversation for each turn length.
# For instance if the conversation has three turns, it will create three conversations.
# Conv 1: turn 1
# Conv 2: turn 1, turn 2
# Conv 3: turn 1, turn 2, turn 3
for t in tqdm(train):
    annotation = t['annotation']
    data = annotation['amt_pre_text'] + "\n" + annotation['amt_table'] + "\n" + annotation['amt_post_text']
    conversation = []
    for i, qa in enumerate(zip(annotation['dialogue_break'], annotation['exe_ans_list'])):
        q, a = qa
        if i == 0:
            user_message = data + "\n\n" + q
        else:
            user_message = q

        conversation.append({
            "role": "user",
            "content": str(user_message)
        })
        conversation.append({
            "role": "assistant",
            "content": str(a)
        })
    dataset.append(conversation)
    file_names.append(t['filename'])


df = pd.DataFrame({'messages': dataset, 'file_names':file_names}, columns=['messages', 'file_names'])
df = df.astype({'messages': 'str', 'file_names': 'str'})

data_folder = 'data/processed/'

os.makedirs(data_folder, exist_ok=True)

file_name = os.path.basename(args.input).replace('.json', '.parquet')
full_path = os.path.join(data_folder, file_name)
print(f"Saving to {full_path}")
df.to_parquet(full_path)


