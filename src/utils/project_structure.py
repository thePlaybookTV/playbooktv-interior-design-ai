"""Project structure creation utilities"""

from pathlib import Path
import shutil
from typing import Dict, Any, Union


def create_structure(base_path: Union[str, Path], structure: Dict[str, Any]):
    """
    Recursively create folder structure from a nested dictionary.
    
    Args:
        base_path: Base path where structure will be created
        structure: Nested dictionary representing folder structure
                  - Keys are folder/file names
                  - Values can be:
                    - dict: nested folders
                    - list: empty folder
                    - str: file contents
    
    Example:
        structure = {
            'src': {
                'data': [],
                'models': {'__init__.py': ''}
            },
            'tests': []
        }
        create_structure('.', structure)
    """
    base_path = Path(base_path)
    
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(exist_ok=True, parents=True)
            create_structure(path, content)
        elif isinstance(content, list):
            path.mkdir(exist_ok=True, parents=True)
        else:
            # String content means it's a file
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(str(content))


def create_default_project_structure(base_path: Union[str, Path] = '.'):
    """
    Create the default PlaybookTV Interior Design AI project structure.
    
    Args:
        base_path: Base path where structure will be created
    """
    structure = {
        'src': {
            'data_collection': {'__init__.py': ''},
            'processing': {'__init__.py': ''},
            'models': {'__init__.py': ''},
            'utils': {'__init__.py': ''}
        },
        'notebooks': {},
        'data': {'.gitkeep': ''},
        'models': {'.gitkeep': ''},
        'docs': {},
        'tests': {},
        'config': {}
    }
    
    create_structure(base_path, structure)
    print(f"✅ Project structure created at {Path(base_path).absolute()}")
