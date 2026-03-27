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

# ============================================================================
# EQUATION TRANSFORMATION ANALYSIS
# ============================================================================
print("=" * 80)
print("  EQUATION TRANSFORMATION — Deep Analysis")
print("=" * 80)

eq_rows = [r for r in rows if r['puzzle_type'] == 'Equation Transformation'][:10]

for i, row in enumerate(eq_rows):
    prompt = row['prompt']
    answer = row['answer']
    
    # Extract examples
    examples = re.findall(r'(.+?)\s*=\s*(.+)', prompt)
    # Filter out the question line and preamble
    clean_examples = []
    for lhs, rhs in examples:
        lhs = lhs.strip()
        rhs = rhs.strip()
        if 'wonderland' in lhs.lower() or 'determine' in lhs.lower() or 'transformation' in lhs.lower():
            continue
        clean_examples.append((lhs, rhs))
    
    # Extract query
    query_m = re.search(r'determine the result for:\s*(.+?)$', prompt, re.MULTILINE)
    query = query_m.group(1).strip() if query_m else "???"
    
    print(f"\n--- Example {i+1} ---")
    print(f"Examples:")
    for lhs, rhs in clean_examples:
        print(f"  '{lhs}' = '{rhs}'")
    print(f"Query: '{query}' = '{answer}'")
    
    # Check if these are numeric or symbolic
    has_numbers = any(re.search(r'\d', lhs) for lhs, _ in clean_examples)
    has_operator = any(re.search(r'[+\-*/|\\^]', lhs) for lhs, _ in clean_examples)
    
    if has_numbers and has_operator:
        print(f"  TYPE: Numeric with operators")
        # Try to parse as actual math with a twist
        for lhs, rhs in clean_examples:
            # Parse operands and operator
            m = re.match(r'(\d+)\s*([+\-*/|\\^])\s*(\d+)', lhs)
            if m:
                a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
                # Try various operations
                results = {
                    '+': a + b, '-': a - b, '*': a * b,
                    'a-b': a - b, 'b-a': b - a,
                    'concat': int(str(a) + str(b)),
                    'a*b': a * b,
                    'digits_a+digits_b': None,
                }
                # Per-digit operations
                sa, sb = str(a).zfill(len(str(b))), str(b).zfill(len(str(a)))
                maxlen = max(len(str(a)), len(str(b)))
                sa2, sb2 = str(a).zfill(maxlen), str(b).zfill(maxlen)
                digit_add = ''.join(str((int(da) + int(db)) % 10) for da, db in zip(sa2, sb2))
                digit_sub = ''.join(str((int(da) - int(db)) % 10) for da, db in zip(sa2, sb2))
                digit_xor = ''.join(str(int(da) ^ int(db)) for da, db in zip(sa2, sb2))
                digit_mul = ''.join(str((int(da) * int(db)) % 10) for da, db in zip(sa2, sb2))
                
                print(f"    {a} {op} {b} = {rhs}")
                print(f"      actual {op}: ", end="")
                if op == '+': print(a + b)
                elif op == '-': print(a - b)
                elif op == '*': print(a * b)
                elif op == '/': print(f"{a/b:.2f}" if b != 0 else "div0")
                elif op == '|': print(a | b)
                elif op == '\\': print(f"a\\b (no standard meaning)")
                elif op == '^': print(a ^ b)
                
                print(f"      digit_add: {digit_add}, digit_sub: {digit_sub}, digit_xor: {digit_xor}, digit_mul: {digit_mul}")
                print(f"      concat(digits): {sa2}{sb2}")
                
                # Check which matches
                for name, val in [('digit_add', digit_add), ('digit_sub', digit_sub), 
                                  ('digit_xor', digit_xor), ('digit_mul', digit_mul)]:
                    if val == rhs:
                        print(f"      *** MATCH: {name} ***")
    else:
        print(f"  TYPE: Symbolic (special characters)")
        # For symbolic: check if it's character-level substitution
        # Check lengths
        for lhs, rhs in clean_examples:
            # Remove operator if present
            parts = re.split(r'([+\-*/|\\^&!@#$%])', lhs)
            print(f"    len('{lhs}')={len(lhs)}, len('{rhs}')={len(rhs)}")


# ============================================================================
# BIT MANIPULATION ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("  BIT MANIPULATION — Deep Analysis")
print("=" * 80)

bit_rows = [r for r in rows if r['puzzle_type'] == 'Bit Manipulation'][:10]

for i, row in enumerate(bit_rows):
    prompt = row['prompt']
    answer = row['answer']
    
    # Extract input->output pairs
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query_m = re.search(r'output for:\s*([01]{8})', prompt)
    query = query_m.group(1) if query_m else "???"
    
    print(f"\n--- Example {i+1} ---")
    print(f"Query: {query} -> {answer}")
    print(f"Training pairs ({len(pairs)}):")
    
    # Test various operations on each pair
    ops_match = {
        'NOT': 0, 'ROT_L1': 0, 'ROT_L2': 0, 'ROT_R1': 0, 'ROT_R2': 0,
        'REVERSE': 0, 'SWAP_NIBBLES': 0, 'XOR_const': {},
    }
    
    for inp, out in pairs:
        a = int(inp, 2)
        b = int(out, 2)
        
        # NOT
        if b == (~a & 0xFF):
            ops_match['NOT'] += 1
        
        # Rotate left by N
        for n in range(1, 8):
            rotated = ((a << n) | (a >> (8 - n))) & 0xFF
            key = f'ROT_L{n}'
            ops_match.setdefault(key, 0)
            if rotated == b:
                ops_match[key] += 1
        
        # Rotate right by N
        for n in range(1, 8):
            rotated = ((a >> n) | (a << (8 - n))) & 0xFF
            key = f'ROT_R{n}'
            ops_match.setdefault(key, 0)
            if rotated == b:
                ops_match[key] += 1
        
        # Reverse bits
        rev = int(inp[::-1], 2)
        if rev == b:
            ops_match['REVERSE'] += 1
        
        # Swap nibbles
        swapped = ((a >> 4) | ((a & 0xF) << 4)) & 0xFF
        if swapped == b:
            ops_match['SWAP_NIBBLES'] += 1
        
        # XOR with constant
        xor_val = a ^ b
        ops_match['XOR_const'][xor_val] = ops_match['XOR_const'].get(xor_val, 0) + 1
        
        # Shift left + something
        for n in range(1, 8):
            shifted = (a << n) & 0xFF
            if shifted == b:
                key = f'SHL{n}'
                ops_match.setdefault(key, 0)
                ops_match[key] += 1
            shifted = (a >> n) & 0xFF
            if shifted == b:
                key = f'SHR{n}'
                ops_match.setdefault(key, 0)
                ops_match[key] += 1
    
    n_pairs = len(pairs)
    print(f"  Single-op matches (need {n_pairs}/{n_pairs}):")
    for op, count in sorted(ops_match.items()):
        if op == 'XOR_const':
            # Find XOR constant that works for all
            for val, cnt in sorted(count.items(), key=lambda x: -x[1]):
                if cnt >= n_pairs * 0.8:
                    print(f"    XOR 0x{val:02X} ({val:08b}): {cnt}/{n_pairs}")
                    # Verify on query
                    predicted = int(query, 2) ^ val
                    print(f"      Query prediction: {predicted:08b}, actual: {answer}")
        elif isinstance(count, int) and count > 0:
            print(f"    {op}: {count}/{n_pairs}")
    
    # Try bit reversal (reverse the entire string)
    # Try per-bit operations looking at bit position mapping
    print(f"  Bit position analysis:")
    # Check if output bit j always comes from input bit k
    for out_bit in range(8):
        sources = set()
        for inp, out in pairs:
            out_val = int(out[7-out_bit])
            for in_bit in range(8):
                in_val = int(inp[7-in_bit])
                # Could this output bit come from this input bit (possibly inverted)?
            # Check: is out_bit always = inp[k] for some k?
            for in_bit in range(8):
                match_normal = all(int(out[7-out_bit]) == int(inp[7-in_bit]) for inp, out in pairs)
                match_inverted = all(int(out[7-out_bit]) == (1 - int(inp[7-in_bit])) for inp, out in pairs)
                if match_normal:
                    sources.add(f"bit{in_bit}")
                if match_inverted:
                    sources.add(f"~bit{in_bit}")
        if sources:
            print(f"    out_bit[{out_bit}] <- {sources}")
        else:
            print(f"    out_bit[{out_bit}] <- COMPLEX (no single-bit source)")
