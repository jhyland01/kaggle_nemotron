"""Test if bit manipulation puzzles can be solved by finding per-bit mappings.
Each output bit might come from:
- A single input bit (possibly inverted)
- OR/AND/XOR of two input bits
- Majority of 3 bits
etc.
"""
import csv
import re
from itertools import combinations

def classify_puzzle(prompt):
    prompt_lower = prompt.lower()
    if re.search(r'bit.?manipul|binary|8.?bit|bitwise|bit.*transform', prompt_lower):
        return 'Bit Manipulation'
    return None

rows = []
with open('train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if classify_puzzle(row['prompt']):
            rows.append(row)

def parse_bit_puzzle(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query_m = re.search(r'output for:\s*([01]{8})', prompt)
    query = query_m.group(1) if query_m else None
    return pairs, query

def get_bit(s, pos):
    """Get bit at position (0=MSB, 7=LSB) from binary string."""
    return int(s[pos])

def solve_bit_permutation(pairs, query):
    """Try to find a per-output-bit mapping.
    For each output bit position, find a function of input bits that works.
    Functions tried:
    - Single bit: in[j] or ~in[j]
    - Two-bit: in[j] XOR in[k], in[j] AND in[k], in[j] OR in[k]
    """
    n = len(pairs)
    if n == 0:
        return None, None
    
    bit_funcs = [None] * 8  # For each output bit position
    
    for out_pos in range(8):
        found = False
        expected = [get_bit(out, out_pos) for _, out in pairs]
        
        # Try single bit (direct or inverted)
        for in_pos in range(8):
            direct = [get_bit(inp, in_pos) for inp, _ in pairs]
            inverted = [1 - b for b in direct]
            
            if direct == expected:
                bit_funcs[out_pos] = ('direct', in_pos)
                found = True
                break
            elif inverted == expected:
                bit_funcs[out_pos] = ('not', in_pos)
                found = True
                break
        
        if found:
            continue
        
        # Try two-bit combinations
        for i, j in combinations(range(8), 2):
            bits_i = [get_bit(inp, i) for inp, _ in pairs]
            bits_j = [get_bit(inp, j) for inp, _ in pairs]
            
            # XOR
            xor_result = [a ^ b for a, b in zip(bits_i, bits_j)]
            if xor_result == expected:
                bit_funcs[out_pos] = ('xor', i, j)
                found = True
                break
            # XNOR
            xnor_result = [1 - (a ^ b) for a, b in zip(bits_i, bits_j)]
            if xnor_result == expected:
                bit_funcs[out_pos] = ('xnor', i, j)
                found = True
                break
            # AND
            and_result = [a & b for a, b in zip(bits_i, bits_j)]
            if and_result == expected:
                bit_funcs[out_pos] = ('and', i, j)
                found = True
                break
            # NAND
            nand_result = [1 - (a & b) for a, b in zip(bits_i, bits_j)]
            if nand_result == expected:
                bit_funcs[out_pos] = ('nand', i, j)
                found = True
                break
            # OR
            or_result = [a | b for a, b in zip(bits_i, bits_j)]
            if or_result == expected:
                bit_funcs[out_pos] = ('or', i, j)
                found = True
                break
            # NOR
            nor_result = [1 - (a | b) for a, b in zip(bits_i, bits_j)]
            if nor_result == expected:
                bit_funcs[out_pos] = ('nor', i, j)
                found = True
                break
        
        if found:
            continue
        
        # Try three-bit majority / choice functions
        for i, j, k in combinations(range(8), 3):
            bits_i = [get_bit(inp, i) for inp, _ in pairs]
            bits_j = [get_bit(inp, j) for inp, _ in pairs]
            bits_k = [get_bit(inp, k) for inp, _ in pairs]
            
            # Majority (2 of 3)
            maj = [1 if (a + b + c) >= 2 else 0 for a, b, c in zip(bits_i, bits_j, bits_k)]
            if maj == expected:
                bit_funcs[out_pos] = ('maj', i, j, k)
                found = True
                break
            # Minority
            minor = [1 if (a + b + c) < 2 else 0 for a, b, c in zip(bits_i, bits_j, bits_k)]
            if minor == expected:
                bit_funcs[out_pos] = ('min3', i, j, k)
                found = True
                break
            # Choice: if i then j else k
            ch = [b if a == 1 else c for a, b, c in zip(bits_i, bits_j, bits_k)]
            if ch == expected:
                bit_funcs[out_pos] = ('choice', i, j, k)
                found = True
                break
            # Choice inverted: if ~i then j else k
            ch_inv = [b if a == 0 else c for a, b, c in zip(bits_i, bits_j, bits_k)]
            if ch_inv == expected:
                bit_funcs[out_pos] = ('choice_inv', i, j, k)
                found = True
                break
        
        if not found:
            bit_funcs[out_pos] = None
    
    # Check if ALL bits are solved
    if all(f is not None for f in bit_funcs):
        # Apply to query
        result = []
        for out_pos in range(8):
            f = bit_funcs[out_pos]
            if f[0] == 'direct':
                result.append(str(get_bit(query, f[1])))
            elif f[0] == 'not':
                result.append(str(1 - get_bit(query, f[1])))
            elif f[0] == 'xor':
                result.append(str(get_bit(query, f[1]) ^ get_bit(query, f[2])))
            elif f[0] == 'xnor':
                result.append(str(1 - (get_bit(query, f[1]) ^ get_bit(query, f[2]))))
            elif f[0] == 'and':
                result.append(str(get_bit(query, f[1]) & get_bit(query, f[2])))
            elif f[0] == 'nand':
                result.append(str(1 - (get_bit(query, f[1]) & get_bit(query, f[2]))))
            elif f[0] == 'or':
                result.append(str(get_bit(query, f[1]) | get_bit(query, f[2])))
            elif f[0] == 'nor':
                result.append(str(1 - (get_bit(query, f[1]) | get_bit(query, f[2]))))
            elif f[0] == 'maj':
                a, b, c = get_bit(query, f[1]), get_bit(query, f[2]), get_bit(query, f[3])
                result.append(str(1 if (a + b + c) >= 2 else 0))
            elif f[0] == 'min3':
                a, b, c = get_bit(query, f[1]), get_bit(query, f[2]), get_bit(query, f[3])
                result.append(str(1 if (a + b + c) < 2 else 0))
            elif f[0] == 'choice':
                a, b, c = get_bit(query, f[1]), get_bit(query, f[2]), get_bit(query, f[3])
                result.append(str(b if a == 1 else c))
            elif f[0] == 'choice_inv':
                a, b, c = get_bit(query, f[1]), get_bit(query, f[2]), get_bit(query, f[3])
                result.append(str(b if a == 0 else c))
        return ''.join(result), bit_funcs
    
    return None, bit_funcs

# Test on all bit manipulation puzzles
solved = 0
partial = 0
total = 0

for i, row in enumerate(rows[:200]):
    pairs, query = parse_bit_puzzle(row['prompt'])
    answer = row['answer']
    total += 1
    
    predicted, funcs = solve_bit_permutation(pairs, query)
    
    if predicted == answer:
        solved += 1
        if i < 20:
            print(f"  [{i}] SOLVED: {funcs}")
    else:
        # Count how many bits we can explain
        explained = sum(1 for f in funcs if f is not None)
        if explained > 0:
            partial += 1
        if i < 20:
            print(f"  [{i}] {'PARTIAL' if explained > 0 else 'UNSOLVED'} ({explained}/8 bits explained): {funcs}")
            if predicted:
                print(f"       predicted={predicted}, actual={answer}")

print(f"\nBit Permutation Solver (first {total} puzzles):")
print(f"  Fully solved:  {solved}/{total} ({100*solved/total:.0f}%)")
print(f"  Partial:       {partial}/{total} ({100*partial/total:.0f}%)")
print(f"  Unsolved:      {total-solved-partial}/{total} ({100*(total-solved-partial)/total:.0f}%)")
