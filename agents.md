# AI Agent Development Guide

This guide provides instructions for AI agents working with this Python project using `uv` for dependency management.

## Environment Setup

### Prerequisites
- Python 3.8+ installed
- `uv` package manager installed
- Git for version control

### Project Structure
```
llinescan/
├── main.py              # Main strip photography application
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Locked dependency versions
├── readme.md           # User documentation
├── agents.md           # This file - AI agent guide
├── results/            # Output directory for generated files
│   ├── video/          # MJPEG video outputs
│   └── debug/          # Debug analysis outputs
└── .gitignore          # Git ignore patterns
```

## Working with UV

### Initial Setup
```bash
# Sync dependencies from lock file
uv sync

# This creates a virtual environment and installs all dependencies
# The virtual environment is automatically managed by uv
```

### Running the Application
```bash
# Basic usage - uv automatically activates the virtual environment
uv run main.py input_video.avi

# With specific parameters
uv run main.py input_video.avi --threshold 0.01 --video --fps 30 --timestamp

# Debug mode for threshold analysis
uv run main.py input_video.avi --debug
```

### Development Workflow

#### Adding Dependencies
```bash
# Add a new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Add with version constraint
uv add "package_name>=1.0,<2.0"
```

#### Managing Dependencies
```bash
# Update all dependencies
uv sync --upgrade

# Remove a dependency
uv remove package_name

# Show installed packages
uv pip list
```

#### Virtual Environment Management
```bash
# uv automatically manages virtual environments
# No need to manually activate/deactivate

# To run Python directly in the environment
uv run python

# To run any command in the environment
uv run <command>
```

## Development Guidelines for AI Agents

### Code Modifications
1. **Read existing code first**: Use `read_file` to understand current implementation
2. **Test changes incrementally**: Make small changes and test with `uv run`
3. **Follow existing patterns**: Maintain consistency with current code style
4. **Update documentation**: Modify `readme.md` when adding features

### Testing Workflow
```bash
# Test basic functionality
uv run main.py test_video.avi --threshold 0.0 --start 0 --end 100

# Test video generation
uv run main.py test_video.avi --video --fps 10 --timestamp --output test.avi

# Test debug mode
uv run main.py test_video.avi --debug
```

### Common Tasks

#### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in `main.py`
3. Test with various parameters
4. Update documentation in `readme.md`
5. Commit and merge when complete

#### Debugging Issues
1. Use `--debug` mode to analyze video processing
2. Test with small frame ranges (`--start 0 --end 50`)
3. Check output file sizes and dimensions
4. Use `ffprobe` to verify video properties

#### Performance Optimization
1. Profile with small test cases first
2. Monitor memory usage with large videos
3. Test different threshold values for compression
4. Verify output quality vs. processing time

### File I/O Patterns

#### Input Validation
```python
# Check video file exists
video_path = Path(args.video_file)
if not video_path.exists():
    print(f"Error: Video file not found: {video_path}")
    sys.exit(1)
```

#### Output Organization
```python
# Auto-generate output paths
results_dir = Path("results/video" if args.video else "results")
results_dir.mkdir(parents=True, exist_ok=True)
```

#### Error Handling
```python
try:
    # Video processing code
    process_video(...)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

### Dependencies Overview

Key packages used in this project:
- **opencv-python**: Video processing and computer vision
- **numpy**: Numerical operations and array handling
- **matplotlib**: Plotting and graph generation for debug mode
- **pathlib**: Modern path handling (built-in)
- **argparse**: Command-line argument parsing (built-in)

### Environment Variables

No special environment variables required. The project uses:
- Current working directory for relative paths
- `results/` subdirectories for organized output
- Automatic virtual environment management via `uv`

### Troubleshooting

#### Common Issues
1. **Import errors**: Run `uv sync` to ensure dependencies are installed
2. **Permission errors**: Check file/directory permissions
3. **Video codec issues**: Ensure input video is readable by OpenCV
4. **Memory issues**: Use smaller frame ranges for testing

#### Debug Commands
```bash
# Check uv environment
uv pip list

# Verify Python version
uv run python --version

# Test basic imports
uv run python -c "import cv2, numpy, matplotlib; print('All imports successful')"
```

### Best Practices for AI Agents

1. **Always use `uv run`** instead of direct python execution
2. **Test with small datasets** before processing large videos
3. **Check file sizes** to ensure reasonable output
4. **Use relative paths** for portability
5. **Follow existing code patterns** for consistency
6. **Update documentation** when adding features
7. **Commit frequently** with descriptive messages
8. **Test edge cases** (empty videos, single frames, etc.)

### Example Agent Workflow

```bash
# 1. Understand current state
uv run main.py --help

# 2. Test existing functionality
uv run main.py sample.avi --debug

# 3. Make incremental changes
# (edit main.py)

# 4. Test changes
uv run main.py sample.avi --new-feature

# 5. Verify output
ls -la results/

# 6. Commit when working
git add . && git commit -m "Add new feature"
```

This workflow ensures reliable development and testing in the uv-managed environment.