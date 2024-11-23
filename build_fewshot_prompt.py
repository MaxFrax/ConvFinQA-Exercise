import pandas as pd

def format_conversation(conversation):
    conv_str = ''
    for message in conversation:
        conv_str += f"## {message['role']}\n{message['content']}\n"
        
    return conv_str

data = pd.read_parquet('data/processed/train.parquet')
data = data.sample(frac=1).reset_index(drop=True)
prompt_tail = 'You are an assistant tasked with answering financial questions accurately. The user will provide contextual information and a question, and you must respond precisely, relying solely on the conversation history. Provide your answer as a single number only, like in Example 1 and Example 2.'

print(data.head())

data['messages'] = data['messages'].apply(lambda x: eval(x))
data['messages_len'] = data['messages'].apply(lambda x: len(x))

print(data['messages_len'].describe())

while True:
    max_len_message = data.loc[data['messages_len'].idxmax(), 'messages']
    min_len_message = data.loc[data['messages_len'].idxmin(), 'messages']
    if data.loc[data['messages_len'].idxmax(), 'file_names'] != data.loc[data['messages_len'].idxmin(), 'file_names']:
        break
    data = data.drop(data['messages_len'].idxmin()).reset_index(drop=True)

prompt = f"# Example 1\n{format_conversation(min_len_message)}\n# Example 2\n{format_conversation(max_len_message)}\n{prompt_tail}"

res_path = 'prompts/system_fewshot.txt'

with open(res_path, 'w') as f:
    f.write(prompt)