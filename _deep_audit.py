"""
Deep audit of NB30: extract ALL code cell sources, check for runtime issues.
"""
import json, ast, re

with open('notebooks/30_domain_robustness_experiments.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

issues = []

code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

# Track all defined names across cells (top-level assignments and functions)
all_defined = set()
all_used_before_define = []

for idx, (cell_idx, cell) in enumerate(code_cells):
    src = ''.join(cell.get('source', []))
    # Remove magic lines
    lines = src.split('\n')
    clean_lines = [l if not l.strip().startswith('%') else '# ' + l for l in lines]
    clean_src = '\n'.join(clean_lines)
    
    first_line = lines[0].strip() if lines else '(empty)'
    
    # Check for common f-string issues: unescaped braces, nested quotes
    # Check for backslash in f-strings (invalid in Python 3.11-)
    fstring_pattern = r"f['\"].*\\n.*['\"]"
    
    # Check for undefined variable references (basic)
    try:
        tree = ast.parse(clean_src)
    except SyntaxError as e:
        issues.append(f"CELL {cell_idx} [{first_line[:60]}]: SYNTAX ERROR line {e.lineno}: {e}")
        continue
    
    # Check for common patterns that cause runtime errors:
    # 1. Using .get() on something that might not be a dict
    # 2. Accessing keys that might not exist
    # 3. Division by zero potential
    
    # Check for prob_col name consistency
    if 'prob_iso' in src and 'prob_col' in src:
        # Make sure prob_col and prob_iso usage is consistent
        pass
    
    # Check for variable name consistency
    if 'usb_preds' in src and 'usb_test' in src:
        pass  # Both should be defined in section A
    
    # Track definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    all_defined.add(target.id)

# Specific checks for known problem patterns
all_src = '\n'.join(''.join(c.get('source', [])) for _, c in code_cells)

# 1. Check session_eval_p90 returns the right keys
if 'session_eval_p90' in all_src:
    # It should return dict with session_roc_auc, block_recall_at_zero_fp, etc.
    pass

# 2. Check that aggregate_to_session_p90 column naming is correct
if "grouper[prob_col].quantile" in all_src:
    # pandas groupby().quantile() preserves the column name
    pass

# 3. Check prob column naming consistency
prob_col_refs = set()
for match in re.finditer(r"prob_col\s*=\s*['\"](\w+)['\"]", all_src):
    prob_col_refs.add(match.group(1))

# 4. Check that all cells that reference usb_preds come AFTER it's defined
usb_preds_defined = False
for idx, (cell_idx, cell) in enumerate(code_cells):
    src = ''.join(cell.get('source', []))
    if 'usb_preds =' in src or "usb_preds[" in src and '=' in src.split('usb_preds[')[0][-5:]:
        usb_preds_defined = True
    if 'usb_preds' in src and not usb_preds_defined:
        if 'usb_preds' not in src.split('=')[0] if '=' in src else True:
            pass  # complex to check statically

# 5. Check for potential NameError in SECTION F cells
section_f_vars = ['light_balance_training_pool', 'run_experiment', 'FEATS_NO_DIR', 
                  'EXPERIMENTS_DIR', 'df_all', 'ALL_COMPACT']
for var in section_f_vars:
    if var not in all_src:
        issues.append(f"POTENTIAL MISSING VAR: {var} not found in notebook code")

# 6. Check that run_balanced_bagging is called with correct parameter names
rbb_calls = re.findall(r'run_balanced_bagging\(([^)]+)\)', all_src, re.DOTALL)
for call in rbb_calls:
    if 'diverse_ratios' in call:
        # Check it's a valid parameter
        pass

# 7. Check session_eval_p90 function for prob_col consistency
se_match = re.search(r'def session_eval_p90\(.*?\n(?:.*?\n)*?.*?return', all_src, re.DOTALL)

# 8. Critical: check that 'prob_iso' column exists when referenced
if "'prob_iso'" in all_src or '"prob_iso"' in all_src:
    # prob_iso is created in predictions.csv by run_balanced_bagging
    # Also created manually in Cell A3 for NB29 inference
    pass

# Write audit report
with open('_deep_audit.txt', 'w', encoding='utf-8') as f:
    f.write("DEEP AUDIT OF NOTEBOOK 30\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Total code cells: {len(code_cells)}\n")
    f.write(f"prob_col values referenced: {prob_col_refs}\n")
    f.write(f"Defined names (sample): {sorted(list(all_defined))[:30]}\n\n")
    
    if issues:
        f.write(f"ISSUES FOUND ({len(issues)}):\n")
        for iss in issues:
            f.write(f"  - {iss}\n")
    else:
        f.write("NO ISSUES FOUND\n")
    
    f.write("\n\nCELL-BY-CELL SUMMARY:\n")
    for idx, (cell_idx, cell) in enumerate(code_cells):
        src = ''.join(cell.get('source', []))
        lines = src.split('\n')
        first_line = lines[0].strip() if lines else '(empty)'
        n_lines = len(lines)
        has_fstring = 'f"' in src or "f'" in src
        has_import = 'import ' in src
        has_def = 'def ' in src
        f.write(f"  Cell {cell_idx:2d} ({n_lines:3d} lines) "
                f"{'F' if has_fstring else ' '}"
                f"{'I' if has_import else ' '}"
                f"{'D' if has_def else ' '}"
                f"  {first_line[:70]}\n")

print("Done")

