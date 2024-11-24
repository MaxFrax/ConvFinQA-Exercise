import re
import argparse
import pandas as pd

def str_to_float(s):
    # Assuming en locale
    s = s.replace(',', '')
    try:
        return float(s)
    except Exception as e:
        print(e)
        print('Error converting ' + s)
        return float('nan')

argparser = argparse.ArgumentParser()
argparser.add_argument('input', type=str, help='Generations file')

args = argparser.parse_args()

data = pd.read_parquet(args.input)

print(data)

print('Na Count')
print(data['generated_text'].isna().value_counts())
print()

print("Exact Match")
matches = data['generated_text'] == data['reference'].astype(str)
print(matches.value_counts())
print(f'Accuracy: {matches.value_counts()[True] / len(matches)*100:.3f}%')
print()

print('How many numbers does an answer usually contain?')
number_re = r"-?\d+((\s|\.|\,)?\d+)+"

extracted_numbers = []
for gen in data['generated_text']:
    matches = re.finditer(number_re, gen, re.MULTILINE)
    numbers = [match.group(0).replace('\n','') for match in matches]
    extracted_numbers.append(numbers)

data['extracted_numbers'] = extracted_numbers

print(data['extracted_numbers'].apply(len).describe())
print()

# Is there the same number in the string?
print('Is the answer represented in the same way somewhere in the generated text?')

count = 0
for reference, extracted in zip(data['reference'], data['extracted_numbers']):
    if reference in extracted:
        count += 1

print(f'Accuracy: {count / len(data)*100:.3f}%')
print()

# Is there the same number converted to float?
print('Is the same float somewhere in the generated text? (float)')
count = 0
total = 0
for reference, extracted in zip(data['reference'], data['extracted_numbers']):
    try:
        reference = float(reference)
        total += 1
    except Exception as e:
        print(e)
        continue

    if reference in [str_to_float(x) for x in extracted]:
        count += 1

print(f'Accuracy: {count / total*100:.3f}%')
print(f'Total: {total}. References not numbers: {len(data) - total}')
print()

# Note: I'm keeping a decimal since rounding everything to int creates a lot of matching zeros
print('Does the number exists somewhere in the generated text represented with only one decimal?')
count = 0
total = 0

for reference, extracted in zip(data['reference'], data['extracted_numbers']):
    try:
        shortened_ref = f'{float(reference):.1f}'
        total += 1
    except:
        continue

    if shortened_ref in [f'{str_to_float(x):.1}' for x in extracted]:
        count += 1

print(f'Accuracy: {count / total*100:.3f}%')
print(f'Total: {total}. References not numbers: {len(data) - total}')