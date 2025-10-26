# Strip Photography / Slit Photography

A digital implementation of **strip photography** (also called **slit photography**) that captures a two-dimensional image as a sequence of one-dimensional images over time.

**How it works:**
Strip photography records a moving scene over time using a camera that observes a narrow strip rather than the full field. This implementation simulates the technique by extracting the same line position from each video frame and assembling them into a composite image where:
- One axis represents **space** (the slit/line being observed)
- The other axis represents **time** (progression through video frames)

**Visual effects:**
- Moving objects appear as visible shapes in the final image
- Stationary objects (like background) appear as horizontal/vertical stripes
- Object width is inversely proportional to speed (faster = narrower, slower = wider)

## Usage

**Column Mode** - Extract vertical lines (columns) from each frame:
```bash
uv run main.py test1.mkv --xcolumn 100 --output test1_column.png
```

**Row Mode** - Extract horizontal lines (rows) from each frame:
```bash
uv run main.py test1.mkv --yrow 200 --output test1_row.png
```

**Debug Mode** - Analyze changes and generate threshold graph:
```bash
uv run main.py test1.mkv --xcolumn 100 --output analysis --debug
```

**Custom Threshold** - Control compression by setting change threshold:
```bash
uv run main.py test1.mkv --xcolumn 100 --output test1_column.png --threshold 0.01
```

## Configure
We use uv to handle pip dependencies. Install with:
```bash
uv sync
```

## Output
- **Column mode**: Extracts vertical line at x-coordinate from each frame
  - Output dimensions: `(source_height, total_frames, 3)`
  - Width = number of frames, Height = source video height
- **Row mode**: Extracts horizontal line at y-coordinate from each frame
  - Output dimensions: `(total_frames, source_width, 3)`
  - Width = source video width, Height = number of frames

Each column/row in the output represents one frame from the input video, showing motion over time.

## Advanced Features

### Change Detection & Compression
The implementation includes intelligent change detection that discards frames with minimal visual changes, creating more compact outputs that focus on motion:

- **Automatic filtering**: Only frames with significant changes are included
- **Configurable threshold**: Use `--threshold` (0-1) to control sensitivity
- **Compression stats**: Shows how many frames were kept vs. skipped

### Debug Mode
Use `--debug` to analyze your video and determine optimal threshold values:

```bash
uv run main.py video.mp4 --xcolumn 500 --output analysis --debug
```

This generates:
- **Change graph**: Visual plot of frame-to-frame changes over time
- **Statistics**: Mean, max, min, and standard deviation of changes
- **Threshold suggestions**: Recommended values with compression ratios

### Threshold Selection Guide
- **0.001-0.005**: High sensitivity, keeps most motion (10-30% compression)
- **0.005-0.02**: Medium sensitivity, good balance (30-70% compression)
- **0.02-0.1**: Low sensitivity, only major changes (70-95% compression)
- **>0.1**: Very low sensitivity, minimal frames (95%+ compression)

### Examples with Compression

Extract with 75% compression (recommended starting point):
```bash
uv run main.py video.mp4 --xcolumn 320 --output compressed.png --threshold 0.01
```

Maximum compression for detecting only major scene changes:
```bash
uv run main.py video.mp4 --xcolumn 320 --output minimal.png --threshold 0.05
```