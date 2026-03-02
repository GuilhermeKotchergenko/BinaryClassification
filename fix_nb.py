import json

with open('notebooks/02_global_execution.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            if 'X, y, test_size=0.2, stratify=y, random_state=42' in line:
                if ')' not in line:
                    new_source.append('            X, y, test_size=0.2, stratify=y, random_state=42\n')
                    new_source.append('        )\n')
                else:
                    new_source.append(line)
            elif '        )\n' == line:
                pass # remove rogue closing brace
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('notebooks/02_global_execution.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

