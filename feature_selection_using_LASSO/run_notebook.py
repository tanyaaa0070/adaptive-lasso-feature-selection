"""Execute the Adaptive LASSO notebook programmatically."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Read the notebook
with open('adaptive_lasso_feature_selection.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Execute
ep = ExecutePreprocessor(timeout=300, kernel_name='python3')
print("Executing notebook... this may take a minute.")
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error during execution: {e}")
    raise

# Save the executed notebook (with outputs)
with open('adaptive_lasso_feature_selection.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Executed notebook saved with outputs.")
