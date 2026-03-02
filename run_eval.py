import json

with open('notebooks/02_global_execution.ipynb', 'r') as f:
    nb = json.load(f)

lines = ["import matplotlib\n", "matplotlib.use('Agg')\n"]

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for line in cell.get('source', []):
            if 'plt.show()' in line or 'plt.plot(' in line or 'plt.figure(' in line or 'plt.xlabel' in line or 'plt.ylabel' in line or 'plt.title' in line or 'plt.legend()' in line or 'plt.grid(' in line or 'plt.subplots(' in line or 'plt.suptitle(' in line or 'plt.tight_layout()' in line or 'axes[' in line or 'ConfusionMatrixDisplay.from_predictions' in line or 'from sklearn.metrics' in line and 'ConfusionMatrixDisplay' in line:
                pass
            else:
                lines.append(line)
        lines.append('\n')

with open('evaluate_all.py', 'w') as f:
    f.writelines(lines)

