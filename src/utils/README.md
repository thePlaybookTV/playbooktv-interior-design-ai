# Utilities Module

This module contains utility functions and helpers for the PlaybookTV Interior Design AI pipeline.

## Modules

### `embeddings.py`
CLIP image embedding utilities.

**Functions:**
- `embed_image_pil(img, device=None, clip_model=None, clip_processor=None)`: Generate L2-normalized CLIP image embeddings

**Example:**
```python
from src.utils import embed_image_pil
from PIL import Image

img = Image.open("path/to/image.jpg")
embedding = embed_image_pil(img)
```

### `setup.py`
Setup and dependency installation utilities.

**Functions:**
- `install_dependencies()`: Install YOLO and SAM2 dependencies

**Example:**
```python
from src.utils import install_dependencies

install_dependencies()
```

### `visualization.py`
Visualization utilities for interior design analysis.

**Functions:**
- `visualize_pristine(db_path, num_samples=6, output_path=None, show_plot=True)`: Visualize images with SAM2 masks and bounding boxes

**Example:**
```python
from src.utils import visualize_pristine

visualize_pristine(
    db_path="./data/metadata.duckdb",
    num_samples=6,
    output_path="visualization.png",
    show_plot=False
)
```

### `project_structure.py`
Project structure creation utilities.

**Functions:**
- `create_structure(base_path, structure)`: Recursively create folder structure from nested dictionary
- `create_default_project_structure(base_path='.')`: Create default PlaybookTV project structure

**Example:**
```python
from src.utils import create_default_project_structure

create_default_project_structure(base_path=".")
```

### `helpers.py`
General helper utilities.

**Functions:**
- `generate_image_id(image_path)`: Generate unique ID for image based on content hash
- `ensure_dir(path)`: Ensure a directory exists, creating it if necessary
- `get_file_size_mb(file_path)`: Get file size in megabytes
- `validate_image_path(image_path)`: Validate that a path points to a valid image file

**Example:**
```python
from src.utils import generate_image_id, ensure_dir, validate_image_path
from pathlib import Path

# Generate image ID
image_id = generate_image_id("path/to/image.jpg")

# Ensure directory exists
dir_path = ensure_dir("./data/processed")

# Validate image path
is_valid = validate_image_path("path/to/image.jpg")
```

## Usage

Import all utilities:
```python
from src.utils import (
    embed_image_pil,
    install_dependencies,
    visualize_pristine,
    create_default_project_structure,
    generate_image_id,
    ensure_dir,
    get_file_size_mb,
    validate_image_path
)
```

Or import specific modules:
```python
from src.utils.embeddings import embed_image_pil
from src.utils.visualization import visualize_pristine
from src.utils.helpers import generate_image_id
```

