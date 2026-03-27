"""Deeper analysis of Equation Transformation and Bit Manipulation puzzles.
Goal: reverse-engineer the rules well enough to build deterministic solvers."""
import csv
import re
from collections import Counter

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
# BIT MANIPULATION — Exhaustive single-operation + 2-op combo solver
# ============================================================================
print("=" * 80)
print("  BIT MANIPULATION — Solver feasibility test")
print("=" * 80)

bit_rows = [r for r in rows if r['puzzle_type'] == 'Bit Manipulation']

def parse_bit_puzzle(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query_m = re.search(r'output for:\s*([01]{8})', prompt)
    query = query_m.group(1) if query_m else None
    return [(int(i, 2), int(o, 2)) for i, o in pairs], query

def apply_op(val, op_name, param=None):
    """Apply a single operation to an 8-bit value."""
    if op_name == 'NOT':
        return (~val) & 0xFF
    elif op_name == 'ROT_L':
        n = param
        return ((val << n) | (val >> (8 - n))) & 0xFF
    elif op_name == 'ROT_R':
        n = param
        return ((val >> n) | (val << (8 - n))) & 0xFF
    elif op_name == 'SHL':
        return (val << param) & 0xFF
    elif op_name == 'SHR':
        return (val >> param) & 0xFF
    elif op_name == 'XOR':
        return val ^ param
    elif op_name == 'AND':
        return val & param
    elif op_name == 'OR':
        return val | param
    elif op_name == 'REVERSE':
        return int(f'{val:08b}'[::-1], 2)
    elif op_name == 'SWAP_NIBBLES':
        return ((val >> 4) | ((val & 0xF) << 4)) & 0xFF
    elif op_name == 'IDENTITY':
        return val
    return None

# Generate all single operations
def all_single_ops():
    ops = [('NOT', None), ('REVERSE', None), ('SWAP_NIBBLES', None), ('IDENTITY', None)]
    for n in range(1, 8):
        ops.append(('ROT_L', n))
        ops.append(('ROT_R', n))
        ops.append(('SHL', n))
        ops.append(('SHR', n))
    for c in range(256):
        ops.append(('XOR', c))
        ops.append(('AND', c))
        ops.append(('OR', c))
    return ops

single_ops = all_single_ops()

def try_single_ops(pairs):
    """Find single operations that match all pairs."""
    matches = []
    for op_name, param in single_ops:
        if all(apply_op(inp, op_name, param) == out for inp, out in pairs):
            matches.append((op_name, param))
    return matches

def try_two_op_combos(pairs):
    """Find 2-operation combos that match all pairs."""
    # Reduced search: only common single ops (not all 256 XOR/AND/OR constants)
    base_ops = [('NOT', None), ('REVERSE', None), ('SWAP_NIBBLES', None)]
    for n in range(1, 8):
        base_ops.extend([('ROT_L', n), ('ROT_R', n), ('SHL', n), ('SHR', n)])
    # Add a few XOR constants
    for c in range(256):
        base_ops.append(('XOR', c))
    
    matches = []
    for op1_name, op1_param in base_ops:
        for op2_name, op2_param in base_ops:
            if all(apply_op(apply_op(inp, op1_name, op1_param), op2_name, op2_param) == out 
                   for inp, out in pairs):
                matches.append(((op1_name, op1_param), (op2_name, op2_param)))
                if len(matches) > 5:  # Don't need all, just proof it works
                    return matches
    return matches

# Test on all bit manipulation puzzles
single_solved = 0
two_op_solved = 0
unsolved = 0
total_bit = len(bit_rows)

print(f"\nTotal Bit Manipulation puzzles: {total_bit}")
print(f"Testing single-op and 2-op solvers...\n")

for i, row in enumerate(bit_rows[:50]):  # Test first 50
    pairs, query = parse_bit_puzzle(row['prompt'])
    answer = row['answer']
    
    # Try single ops
    matches = try_single_ops(pairs)
    if matches:
        # Verify on query
        op_name, param = matches[0]
        if query:
            predicted = f'{apply_op(int(query, 2), op_name, param):08b}'
            correct = predicted == answer
            if correct:
                single_solved += 1
                if i < 15:
                    print(f"  [{i}] SINGLE-OP SOLVED: {op_name}({param}) → predicted={predicted}, answer={answer} ✓")
                continue
            else:
                if i < 15:
                    print(f"  [{i}] Single op found but wrong prediction: {op_name}({param}) → {predicted} vs {answer}")
    
    # Try 2-op combos
    matches = try_two_op_combos(pairs)
    if matches:
        (op1_name, op1_param), (op2_name, op2_param) = matches[0]
        if query:
            predicted = f'{apply_op(apply_op(int(query, 2), op1_name, op1_param), op2_name, op2_param):08b}'
            correct = predicted == answer
            if correct:
                two_op_solved += 1
                if i < 15:
                    print(f"  [{i}] 2-OP SOLVED: {op1_name}({op1_param}) then {op2_name}({op2_param}) → {predicted} ✓")
                continue
            else:
                if i < 15:
                    print(f"  [{i}] 2-op found but wrong: {matches[0]} → {predicted} vs {answer}")
    
    unsolved += 1
    if i < 15:
        print(f"  [{i}] UNSOLVED (query={query}, answer={answer})")

tested = min(50, total_bit)
print(f"\nResults (first {tested} puzzles):")
print(f"  Single-op solved: {single_solved}/{tested} ({100*single_solved/tested:.0f}%)")
print(f"  Two-op solved:    {two_op_solved}/{tested} ({100*two_op_solved/tested:.0f}%)")
print(f"  Unsolved:         {unsolved}/{tested} ({100*unsolved/tested:.0f}%)")


# ============================================================================
# EQUATION TRANSFORMATION — Pattern analysis
# ============================================================================
print("\n\n" + "=" * 80)
print("  EQUATION TRANSFORMATION — Pattern analysis")
print("=" * 80)

eq_rows = [r for r in rows if r['puzzle_type'] == 'Equation Transformation']

def parse_eq_puzzle(prompt):
    """Extract examples and query from equation transformation prompt."""
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query_m = re.search(r'determine the result for:\s*(.+)', line, re.IGNORECASE)
            if query_m:
                query = query_m.group(1).strip()
        elif ' = ' in line and 'wonderland' not in line.lower() and 'transformation' not in line.lower() and 'examples' not in line.lower():
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
    return examples, query

# Analyze structure of all equation transformation puzzles
numeric_count = 0
symbolic_count = 0
input_lengths = Counter()
output_lengths = Counter()

for row in eq_rows[:100]:
    examples, query = parse_eq_puzzle(row['prompt'])
    answer = row['answer']
    
    for lhs, rhs in examples:
        input_lengths[len(lhs)] += 1
        output_lengths[len(rhs)] += 1
    
    # Check if numeric
    has_digits = any(re.search(r'\d', lhs) for lhs, _ in examples)
    if has_digits:
        numeric_count += 1
    else:
        symbolic_count += 1

print(f"\nTotal Equation Transformation puzzles: {len(eq_rows)}")
print(f"  Numeric (has digits): {numeric_count}/100 sampled")
print(f"  Symbolic (no digits): {symbolic_count}/100 sampled")
print(f"\nInput lengths: {dict(input_lengths.most_common(10))}")
print(f"Output lengths: {dict(output_lengths.most_common(10))}")

# Now test specific theories on symbolic puzzles
print("\n--- Testing theory: output = f(ASCII codes) ---")
for idx, row in enumerate(eq_rows[:20]):
    examples, query = parse_eq_puzzle(row['prompt'])
    answer = row['answer']
    if not examples or not query:
        continue
    
    # All inputs should be 5 chars: A[0]A[1] op B[0]B[1]
    all_5 = all(len(lhs) == 5 for lhs, _ in examples) and len(query) == 5
    if not all_5:
        continue
    
    # Parse as A (2 chars) + op (1 char) + B (2 chars)
    parsed = []
    for lhs, rhs in examples:
        a0, a1, op, b0, b1 = [ord(c) for c in lhs]
        parsed.append({
            'a': (a0, a1), 'op': op, 'b': (b0, b1),
            'op_char': lhs[2], 'rhs': rhs, 'rhs_ords': [ord(c) for c in rhs]
        })
    
    q_a = (ord(query[0]), ord(query[1]))
    q_op = ord(query[2])
    q_b = (ord(query[3]), ord(query[4]))
    
    # Theory 1: For each puzzle, there's a lookup table mapping each char to another
    # Check if the operator char is always the same across examples
    ops_in_puzzle = set(p['op_char'] for p in parsed)
    
    # Theory 2: digit-wise operation with modular arithmetic
    # For each output char, check if it's a simple function of A and B chars
    print(f"\n  Puzzle {idx} (op chars: {ops_in_puzzle}, query_op='{query[2]}'):")
    for p in parsed[:3]:
        print(f"    {chr(p['a'][0])}{chr(p['a'][1])} {p['op_char']} {chr(p['b'][0])}{chr(p['b'][1])} = {p['rhs']}")
    print(f"    Query: {query} = {answer}")
    
    # Check: are all operators the same?
    if len(ops_in_puzzle) == 1:
        print(f"    [Same operator throughout: '{list(ops_in_puzzle)[0]}']")
    else:
        print(f"    [Mixed operators: {ops_in_puzzle}]")
    
    # Check output pattern: does output come from rearranging input chars?
    for p in parsed:
        lhs = chr(p['a'][0]) + chr(p['a'][1]) + p['op_char'] + chr(p['b'][0]) + chr(p['b'][1])
        rhs = p['rhs']
        # Check if every char in output appears in input
        all_from_input = all(c in lhs for c in rhs)
        if all_from_input:
            # Which positions?
            positions = []
            remaining = list(lhs)
            for c in rhs:
                if c in remaining:
                    pos = remaining.index(c)
                    positions.append(pos)
                    remaining[pos] = None
            # print(f"      '{lhs}' → '{rhs}', output positions from input: {positions}")
