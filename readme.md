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