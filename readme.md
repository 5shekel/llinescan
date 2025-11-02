# Strip Photography / Slit Photography

A digital implementation of **strip photography** (also called **slit photography**) that captures a two-dimensional image as a sequence of one-dimensional images over time.

![demo](demo2.jpg)

Strip photography records a moving scene over time by extracting the same line position from each video frame and assembling them into a composite image where:
- One axis represents **space** (the slit/line being observed)
- The other axis represents **time** (progression through video frames)
- Moving objects appear as visible shapes, stationary objects as stripes
- Object width is inversely proportional to speed (faster = narrower, slower = wider)

## Usage

**Basic Usage** - Uses smart defaults (yrow=8, auto output to results/):
```bash
uv run main.py .\line500fps32pix.mp4
```
Output: `results/line500fps32pix_a3f2_t0_01.jpg` (auto-generated with UUID and threshold)

**Row Mode** - Extract horizontal lines with custom settings:
```bash
uv run main.py .\line500fps32pix.mp4 --yrow 16 --threshold 0.005
```

**With Relax** - Include extra frames around threshold frames for smoother transitions:
```bash
uv run main.py .\line500fps32pix.mp4 --relax 5
# or use default relax value (100):
uv run main.py .\line500fps32pix.mp4 --relax
```

**Debug Mode** - Analyze changes and generate threshold recommendations:
```bash
uv run main.py .\line500fps32pix.mp4 --debug
```
Output: `results/debug/line500fps32pix_7c91_t0_01_changes.png`
![debug view](demo_changes2.jpg)

Debug mode generates PowerShell commands to test all suggested thresholds:
```powershell
0.0087,0.0098,0.0112,0.0121,0.0159 | %{uv run .\main.py video.mp4 --threshold $_}
```

## Setup
Install dependencies with uv:
```bash
uv sync
```

## Parameters

### Required
- `video_file` - Path to input video file

### Optional
- `--xcolumn N` - Extract vertical line at x-coordinate (column mode)
- `--yrow N` - Extract horizontal line at y-coordinate (row mode, **default: 8**)
- `--output PATH` - Output file path (default: auto-generated to `results/` with UUID)
- `--threshold N` - Change threshold 0-1 for frame inclusion (default: 0.01)
- `--relax [N]` - Include N extra frames before/after threshold frames (default: 0, or 100 if flag used without value)
- `--debug` - Analyze changes without creating strip image, outputs to `results/debug/`

### Output Modes
- **Column mode**: Extracts vertical line (`--xcolumn`) → Width = frames, Height = video height
- **Row mode**: Extracts horizontal line (`--yrow`, **default**) → Rotated 90° CW for proper orientation

## Features

**Smart Defaults**:
- Defaults to row mode at y=8 if no mode specified
- Auto-generates output filename with UUID and threshold: `results/video_a3f2_t0_01.jpg`
- Automatically appends `.jpg` extension if none provided
- Row mode output is rotated 90° clockwise for proper viewing

**Change Detection**: Automatically filters frames with minimal changes using configurable thresholds
- Use `--threshold` (0-1) to control sensitivity
- `--debug` mode provides threshold recommendations and change analysis
- Higher thresholds = more compression, fewer frames

**Relax Feature**: Include extra frames around significant changes for smoother transitions
- `--relax 5` includes 5 frames before/after each threshold frame
- `--relax` without value defaults to 100 frames

**Threshold Guide**:
- `0.001-0.005`: High sensitivity (10-30% compression)
- `0.005-0.02`: Medium sensitivity (30-70% compression)
- `0.02+`: Low sensitivity (70%+ compression)

**Output Organization**:
- Normal mode: `results/` folder
- Debug mode: `results/debug/` folder
