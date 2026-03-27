import csv
import re

def classify_puzzle(prompt):
    prompt_lower = prompt.lower()
    if re.search(r'numeral system|base[- ]?\d|number.*convert|radix|secret number', prompt_lower):
        return 'Number Base Conversion'
    elif re.search(r'gravit|gravity|falling|free.?fall|acceleration due to', prompt_lower):
        return 'Gravitational Constant'
    elif re.search(r'transformation rule|equation.*transform|secret.*rule.*equation|rule.*applied.*equation', prompt_lower):
        return 'Equation Transformation'
    elif re.search(r'encrypt|cipher|secret.*code.*letter|coded.*message|secret.*text', prompt_lower):
        return 'Text Encryption'
    elif re.search(r'bit.?manipul|binary|8.?bit|bitwise|bit.*transform', prompt_lower):
        return 'Bit Manipulation'
    elif re.search(r'unit.?conver|measurement|becomes.*\d|secret.*conver.*measur', prompt_lower):
        return 'Unit Conversion'
    else:
        return 'Unknown'

rows = []
with open('train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['puzzle_type'] = classify_puzzle(row['prompt'])
        rows.append(row)

for ptype in ['Equation Transformation', 'Bit Manipulation']:
    subset = [r for r in rows if r['puzzle_type'] == ptype][:10]
    print(f"\n{'='*80}")
    print(f"  {ptype} -- {len(subset)} examples")
    print(f"{'='*80}")
    for i, row in enumerate(subset):
        print(f"\n--- Example {i+1} (id={row['id']}) ---")
        print(f"PROMPT:\n{row['prompt']}")
        print(f"\nANSWER: {row['answer']}")
        print()
