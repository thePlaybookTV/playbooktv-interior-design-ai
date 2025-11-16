# AI Agent Guidelines - PlaybookTV Interior Design AI

## Commands
- **Test**: `pytest tests/` (all tests) or `pytest path/to/test_file.py` (single test)
- **API Server**: `./start_api.sh` or `uvicorn api.main:app --reload`
- **Training**: `python scripts/run_phase2_training.py --db database_metadata.duckdb --output ./phase2_outputs`
- **Modal Deploy**: `modal deploy modal_functions/sd_inference_complete.py`
- **Lint**: `flake8` (configured in requirements.txt)
- **Format**: `black .`

## Architecture
- **Tech Stack**: Python 3.9+, PyTorch, FastAPI, DuckDB, Modal (serverless GPU)
- **Structure**: `src/` (models, processing, data_collection, utils), `api/` (FastAPI server), `scripts/` (training), `modal_functions/` (GPU inference)
- **Database**: DuckDB at `database_metadata.duckdb` (tables: `images`, `furniture_detections`)
- **Models**: YOLOv8 + SAM2 (object detection), ResNet/EfficientNet/ViT (style/room classification), ensemble classifiers
- **Data**: 294 furniture categories, 7 room types, 9 design styles (defined in `interior_taxonomy.py`)

## Code Style
- **Imports**: Standard library → third-party → local modules, use absolute imports from `src/`
- **Types**: Use type hints for function signatures, prefer explicit over implicit
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use logging module (configured), catch specific exceptions, provide context in error messages
- **Path Handling**: Use `Path` from `pathlib`, support absolute paths, handle missing files gracefully
- **Codacy**: MUST run `codacy_cli_analyze` after editing files (see `.cursor/rules/codacy.mdc`)
