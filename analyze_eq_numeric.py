"""Test the theory that equation transformation puzzles use:
- 5-char input: A[0]A[1] + operator + B[0]B[1]
- Numeric puzzles: each operator char maps to a standard math operation
- Symbolic puzzles: ASCII-code arithmetic with the same structure
"""
import csv
import re
from itertools import product

def classify_puzzle(prompt):
    prompt_lower = prompt.lower()
    if re.search(r'transformation rule|equation.*transform|secret.*rule.*equation|rule.*applied.*equation', prompt_lower):
        return 'Equation Transformation'
    elif re.search(r'bit.?manipul|binary|8.?bit|bitwise|bit.*transform', prompt_lower):
        return 'Bit Manipulation'
    return None

rows = []
with open('train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pt = classify_puzzle(row['prompt'])
        if pt:
            row['puzzle_type'] = pt
            rows.append(row)

def parse_eq_puzzle(prompt):
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

# ============================================================================
# NUMERIC EQUATION TRANSFORM — Each operator maps to +, -, *, concat, etc.
# ============================================================================
print("=" * 80)
print("  NUMERIC EQUATION TRANSFORMS — Operator Discovery")
print("=" * 80)

eq_rows = [r for r in rows if r['puzzle_type'] == 'Equation Transformation']

OPERATIONS = {
    'add': lambda a, b: str(a + b),
    'sub_ab': lambda a, b: str(a - b),
    'sub_ba': lambda a, b: str(b - a),
    'abs_sub': lambda a, b: str(abs(a - b)),
    'mul': lambda a, b: str(a * b),
    'concat_ab': lambda a, b: str(a) + str(b),
    'concat_ba': lambda a, b: str(b) + str(a),
    'div_ab': lambda a, b: str(a // b) if b != 0 else None,
    'div_ba': lambda a, b: str(b // a) if a != 0 else None,
    'mod_ab': lambda a, b: str(a % b) if b != 0 else None,
    'mod_ba': lambda a, b: str(b % a) if a != 0 else None,
    'pow_ab': lambda a, b: str(a ** b) if b < 10 else None,
    'pow_ba': lambda a, b: str(b ** a) if a < 10 else None,
    'xor': lambda a, b: str(a ^ b),
    'and': lambda a, b: str(a & b),
    'or': lambda a, b: str(a | b),
    'max': lambda a, b: str(max(a, b)),
    'min': lambda a, b: str(min(a, b)),
    'digit_add': lambda a, b: ''.join(str((int(da) + int(db)) % 10) for da, db in zip(str(a).zfill(max(len(str(a)), len(str(b)))), str(b).zfill(max(len(str(a)), len(str(b)))))),
    'digit_sub': lambda a, b: ''.join(str((int(da) - int(db)) % 10) for da, db in zip(str(a).zfill(max(len(str(a)), len(str(b)))), str(b).zfill(max(len(str(a)), len(str(b)))))),
    'digit_mul': lambda a, b: ''.join(str((int(da) * int(db)) % 10) for da, db in zip(str(a).zfill(max(len(str(a)), len(str(b)))), str(b).zfill(max(len(str(a)), len(str(b)))))),
}

numeric_solved = 0
numeric_total = 0

for idx, row in enumerate(eq_rows[:200]):
    examples, query = parse_eq_puzzle(row['prompt'])
    answer = row['answer']
    if not examples or not query or len(query) != 5:
        continue
    
    # Check if numeric: left and right operands are digits
    all_numeric = True
    parsed_examples = []
    for lhs, rhs in examples:
        if len(lhs) != 5:
            all_numeric = False
            break
        left_s, op_char, right_s = lhs[:2], lhs[2], lhs[3:]
        if not left_s.isdigit() or not right_s.isdigit():
            all_numeric = False
            break
        parsed_examples.append((int(left_s), op_char, int(right_s), rhs))
    
    if not all_numeric or not parsed_examples:
        continue
    
    numeric_total += 1
    
    # Group by operator
    ops_by_char = {}
    for left, op_char, right, rhs in parsed_examples:
        if op_char not in ops_by_char:
            ops_by_char[op_char] = []
        ops_by_char[op_char].append((left, right, rhs))
    
    # For each operator char, find which operation matches
    op_mapping = {}
    for op_char, pairs in ops_by_char.items():
        for op_name, op_func in OPERATIONS.items():
            matches = True
            for left, right, rhs in pairs:
                try:
                    result = op_func(left, right)
                    if result != rhs:
                        matches = False
                        break
                except:
                    matches = False
                    break
            if matches:
                op_mapping[op_char] = op_name
                break
    
    # Apply to query
    q_left, q_op, q_right = int(query[:2]), query[2], int(query[3:])
    
    if q_op in op_mapping:
        op_name = op_mapping[q_op]
        try:
            predicted = OPERATIONS[op_name](q_left, q_right)
            if predicted == answer:
                numeric_solved += 1
                if idx < 40:
                    print(f"  [{idx}] SOLVED: op_mapping={op_mapping}, {q_left} '{q_op}'({op_name}) {q_right} = {predicted} ✓")
            else:
                if idx < 40:
                    print(f"  [{idx}] WRONG: op_mapping={op_mapping}, predicted={predicted}, answer={answer}")
        except:
            if idx < 40:
                print(f"  [{idx}] ERROR applying {op_name}")
    else:
        if idx < 40:
            # Try to find the mapping from the query + answer
            print(f"  [{idx}] Query op '{q_op}' not in examples. Examples: {[(left, op_c, right, rhs) for left, op_c, right, rhs in parsed_examples[:3]]}")
            print(f"         op_mapping so far: {op_mapping}")
            print(f"         Query: {q_left} '{q_op}' {q_right} = {answer}")

print(f"\nNumeric Equation Transform results (first 200 puzzles):")
print(f"  Numeric puzzles found: {numeric_total}")
print(f"  Solved: {numeric_solved}/{numeric_total} ({100*numeric_solved/numeric_total:.0f}% of numeric)")


# ============================================================================
# SYMBOLIC EQUATION TRANSFORMS — Test ASCII arithmetic theory
# ============================================================================
print("\n\n" + "=" * 80)
print("  SYMBOLIC EQUATION TRANSFORMS — ASCII Arithmetic Test")
print("=" * 80)

# Theory: each char maps to a value (0-based from some offset).
# Operators apply modular arithmetic on these values, producing output chars.
# The char ordering might be a custom permutation, not just ASCII order.

# First, let's see all unique characters that appear
all_chars = set()
for row in eq_rows[:200]:
    examples, query = parse_eq_puzzle(row['prompt'])
    if not examples or not query:
        continue
    for lhs, rhs in examples:
        all_chars.update(lhs)
        all_chars.update(rhs)
    all_chars.update(query)
    all_chars.update(row['answer'])

# Standard printable ASCII chars used
printable_special = sorted(c for c in all_chars if not c.isalnum())
print(f"Special chars seen: {printable_special}")
print(f"Total unique chars: {len(all_chars)}")

# Let's look at a few symbolic puzzles very carefully
# For each, treat chars as their ordinal and check modular arithmetic
print("\n--- Detailed symbolic analysis ---")

sym_solved = 0
sym_total = 0

for idx, row in enumerate(eq_rows[:200]):
    examples, query = parse_eq_puzzle(row['prompt'])
    answer = row['answer']
    if not examples or not query or len(query) != 5:
        continue
    
    # Skip numeric
    if query[:2].isdigit() and query[3:].isdigit():
        continue
    
    sym_total += 1
    
    # Parse all examples
    parsed = []
    for lhs, rhs in examples:
        if len(lhs) != 5:
            continue
        parsed.append({
            'a0': ord(lhs[0]), 'a1': ord(lhs[1]),
            'op': ord(lhs[2]), 'op_char': lhs[2],
            'b0': ord(lhs[3]), 'b1': ord(lhs[4]),
            'rhs': rhs, 'rhs_ords': [ord(c) for c in rhs]
        })
    
    if not parsed:
        continue
    
    # Group by operator char
    by_op = {}
    for p in parsed:
        by_op.setdefault(p['op_char'], []).append(p)
    
    # For each operator, test theories on (a0,a1) op (b0,b1) → rhs
    # Theory: the operator defines a simple per-position operation on ordinals
    # with modular arithmetic in a specific range
    
    # Determine char range
    all_ords = set()
    for p in parsed:
        all_ords.update([p['a0'], p['a1'], p['b0'], p['b1']])
        all_ords.update(p['rhs_ords'])
    min_ord, max_ord = min(all_ords), max(all_ords)
    char_range = max_ord - min_ord + 1
    
    # Test: for each op, is the output a specific arithmetic combo of inputs?
    # The output length varies, suggesting the rule might produce variable-length results
    
    if idx < 25 and sym_total <= 10:
        print(f"\n  Puzzle {idx} (char range [{min_ord}..{max_ord}]={char_range}, ops={set(by_op.keys())}):")
        for p in parsed[:4]:
            a_str = chr(p['a0']) + chr(p['a1'])
            b_str = chr(p['b0']) + chr(p['b1'])
            print(f"    {a_str} {p['op_char']} {b_str} = {p['rhs']}  (ords: [{p['a0']},{p['a1']}] {p['op']} [{p['b0']},{p['b1']}] = {p['rhs_ords']})")
        print(f"    Query: {query} = {answer}  (ords: [{ord(query[0])},{ord(query[1])}] {ord(query[2])} [{ord(query[3])},{ord(query[4])}] = {[ord(c) for c in answer]})")

# ============================================================================
# EQUATION TRANSFORM — Count what fraction we CANNOT solve
# ============================================================================
print("\n\n" + "=" * 80)
print(f"  EQUATION TRANSFORM SUMMARY")
print(f"=" * 80)
print(f"  Total puzzles: {len(eq_rows)}")
print(f"  Numeric: ~{numeric_total} in first 200 ({100*numeric_total/200:.0f}%), solved {numeric_solved}")
print(f"  Symbolic: ~{sym_total} in first 200 ({100*sym_total/200:.0f}%)")
print(f"  → Can deterministically solve: ~{100*numeric_solved/200:.0f}% of puzzles")
print(f"  → Need SFT scaffold for: ~{100*(200-numeric_solved)/200:.0f}% of puzzles")
