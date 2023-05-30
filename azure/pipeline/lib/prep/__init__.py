'''
Re-exporting stuff in `main.py` 

This is so we can import it in `azure/pipeline/main.py` 
without needing to specify `.main` at the end
'''

# dot in front specifies relative imports
from .main import prepare_data_component