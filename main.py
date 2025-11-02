#!/usr/bin/env python3
"""
Strip Photography / Slit Photography Implementation

A digital implementation of strip photography that captures a two-dimensional 
image as a sequence of one-dimensional images over time.
"""

import argparse
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uuid


def calculate_line_difference(line1, line2):
    """
    Calculate the difference between two lines (column or row).
    
    Args:
        line1, line2: numpy arrays representing lines from consecutive frames
        
    Returns:
        float: normalized difference value between 0 and 1
    """
    # Convert to float for calculation
    diff = np.abs(line1.astype(np.float32) - line2.astype(np.float32))
    # Calculate mean difference across all channels
    mean_diff = np.mean(diff)
    # Normalize to 0-255 range
    return mean_diff / 255.0


def generate_change_graph(changes, output_path, threshold=None):
    """
    Generate a graph showing change values over time.
    
    Args:
        changes: List of change values
        output_path: Path for output graph image
        threshold: Optional threshold line to display
    """
    plt.figure(figsize=(12, 6))
    plt.plot(changes, linewidth=1, alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Change Value (0-1)')
    plt.title('Line Change Detection Over Time')
    plt.grid(True, alpha=0.3)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--',
                   label=f'Threshold: {threshold:.3f}')
        plt.legend()
    
    # Add statistics
    mean_change = np.mean(changes)
    max_change = np.max(changes)
    std_change = np.std(changes)
    
    stats_text = f'Mean: {mean_change:.3f}\nMax: {max_change:.3f}\nStd: {std_change:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Change graph saved to: {output_path}")


def analyze_changes_only(video_path, x_column=None, y_row=None, debug_output=None):
    """
    Analyze changes in video without generating strip image.
    Used for debug mode to generate change threshold graphs.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of column to analyze (if column mode)
        y_row: Y-coordinate of row to analyze (if row mode)
        debug_output: Base path for debug outputs
        
    Returns:
        List of change values
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if x_column is not None:
        if x_column >= frame_width:
            raise ValueError(f"Column {x_column} is outside video width ({frame_width})")
        print(f"Analyzing column {x_column} from {frame_width}x{frame_height} frames")
    else:
        if y_row >= frame_height:
            raise ValueError(f"Row {y_row} is outside video height ({frame_height})")
        print(f"Analyzing row {y_row} from {frame_width}x{frame_height} frames")
    
    print(f"Processing {total_frames} frames for change analysis...")
    
    changes = []
    previous_line = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract current line (column or row)
        if x_column is not None:
            current_line = frame[:, x_column, :].copy()
        else:
            current_line = frame[y_row, :, :].copy()
        
        # Calculate change from previous frame
        if previous_line is not None:
            change = calculate_line_difference(current_line, previous_line)
            changes.append(change)
        
        previous_line = current_line
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Analyzed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    if debug_output:
        # Generate change graph (debug_output is now a Path object)
        graph_path = debug_output.parent / f"{debug_output.stem}_changes.png"
        generate_change_graph(changes, graph_path)
        
        # Generate statistics
        if changes:
            print(f"\nChange Analysis Statistics:")
            print(f"Total frames analyzed: {len(changes)}")
            print(f"Mean change: {np.mean(changes):.4f}")
            print(f"Max change: {np.max(changes):.4f}")
            print(f"Min change: {np.min(changes):.4f}")
            print(f"Std deviation: {np.std(changes):.4f}")
            
            # Suggest thresholds
            percentiles = [50, 75, 90, 95, 99]
            threshold_values = []
            print(f"\nSuggested threshold values:")
            for p in percentiles:
                thresh = np.percentile(changes, p)
                threshold_values.append(thresh)
                frames_above = np.sum(np.array(changes) >= thresh)
                compression = (len(changes) - frames_above) / len(changes) * 100
                print(f"  {p}th percentile: {thresh:.4f} (keeps {frames_above} frames, {compression:.1f}% compression)")
            
            # Generate PowerShell command to test all suggested thresholds
            threshold_list = ",".join([f"{t:.4f}" for t in threshold_values])
            video_path_str = str(video_path.absolute())
            pwsh_cmd = f"{threshold_list} | %{{uv run .\\main.py {video_path_str} --threshold $_}}"
            print(f"\nPowerShell command to test all thresholds:")
            print(f"  {pwsh_cmd}")
    
    return changes


def extract_column_strip(video_path, x_column, output_path, change_threshold=0.005, relax=0):
    """
    Extract vertical strip at x_column from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if x_column >= frame_width:
        raise ValueError(f"Column {x_column} is outside video width ({frame_width})")
    
    print(f"Processing {total_frames} frames...")
    print(f"Extracting column {x_column} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all columns and identify significant frames
    all_columns = []
    changes = []
    previous_column = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract current column
        current_column = frame[:, x_column, :].copy()
        all_columns.append(current_column)
        
        # Calculate change from previous frame
        if previous_column is not None:
            change = calculate_line_difference(current_column, previous_column)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_column = current_column
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    # Second pass: determine which frames to include
    include_mask = [False] * len(all_columns)
    
    for i, change in enumerate(changes):
        if i == 0 or change >= change_threshold:
            # Mark this frame and surrounding frames
            start = max(0, i - relax)
            end = min(len(all_columns), i + relax + 1)
            for j in range(start, end):
                include_mask[j] = True
    
    # Collect significant columns
    significant_columns = [col for i, col in enumerate(all_columns) if include_mask[i]]
    included_frames = sum(include_mask)
    skipped_frames = len(all_columns) - included_frames
    
    if not significant_columns:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    # Convert list to numpy array
    strip_image = np.stack(significant_columns, axis=1)
    
    print(f"Original frames: {total_frames}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/total_frames:.1%}")
    print(f"Output dimensions: {strip_image.shape}")
    print(f"Saving to: {output_path}")
    
    # Save the strip image
    cv2.imwrite(str(output_path), strip_image)


def extract_row_strip(video_path, y_row, output_path, change_threshold=0.01, relax=0):
    """
    Extract horizontal strip at y_row from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if y_row >= frame_height:
        raise ValueError(f"Row {y_row} is outside video height ({frame_height})")
    
    print(f"Processing {total_frames} frames...")
    print(f"Extracting row {y_row} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all rows and identify significant frames
    all_rows = []
    changes = []
    previous_row = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract current row
        current_row = frame[y_row, :, :].copy()
        all_rows.append(current_row)
        
        # Calculate change from previous frame
        if previous_row is not None:
            change = calculate_line_difference(current_row, previous_row)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_row = current_row
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    # Second pass: determine which frames to include
    include_mask = [False] * len(all_rows)
    
    for i, change in enumerate(changes):
        if i == 0 or change >= change_threshold:
            # Mark this frame and surrounding frames
            start = max(0, i - relax)
            end = min(len(all_rows), i + relax + 1)
            for j in range(start, end):
                include_mask[j] = True
    
    # Collect significant rows
    significant_rows = [row for i, row in enumerate(all_rows) if include_mask[i]]
    included_frames = sum(include_mask)
    skipped_frames = len(all_rows) - included_frames
    
    if not significant_rows:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    # Convert list to numpy array
    strip_image = np.stack(significant_rows, axis=0)
    
    # Rotate clockwise 90 degrees for row mode
    strip_image = cv2.rotate(strip_image, cv2.ROTATE_90_CLOCKWISE)
    
    print(f"Original frames: {total_frames}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/total_frames:.1%}")
    print(f"Output dimensions: {strip_image.shape} (rotated 90° CW)")
    print(f"Saving to: {output_path}")
    
    # Save the strip image
    cv2.imwrite(str(output_path), strip_image)


def main():
    """Main entry point for the strip photography tool."""
    parser = argparse.ArgumentParser(
        description="Extract strip photography effects from video files"
    )
    
    parser.add_argument(
        "video_file",
        help="Input video file path"
    )
    
    parser.add_argument(
        "--xcolumn",
        type=int,
        help="Extract vertical line at x-coordinate (column mode)"
    )
    
    parser.add_argument(
        "--yrow",
        type=int,
        help="Extract horizontal line at y-coordinate (row mode, default: 8)"
    )
    
    parser.add_argument(
        "--output",
        help="Output image file path (default: results/<input_name>.jpg)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Change threshold (0-1) for including frames (default: 0.01)"
    )
    
    parser.add_argument(
        "--relax",
        type=int,
        nargs='?',
        const=100,
        default=0,
        help="Include N extra frames before/after frames exceeding threshold (default: 0, or 100 if flag used without value)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: analyze changes and generate threshold graph without creating strip image"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Validate mode selection
    if args.xcolumn is not None and args.yrow is not None:
        print("Error: Cannot specify both --xcolumn and --yrow. Choose one mode.")
        sys.exit(1)
    
    # Default to yrow=8 if neither mode specified
    if args.xcolumn is None and args.yrow is None:
        args.yrow = 8
        print(f"Using default: --yrow={args.yrow}")
    
    # Validate coordinates
    if args.xcolumn is not None and args.xcolumn < 0:
        print("Error: --xcolumn must be non-negative")
        sys.exit(1)
        
    if args.yrow is not None and args.yrow < 0:
        print("Error: --yrow must be non-negative")
        sys.exit(1)
    
    # Validate threshold
    if not (0 <= args.threshold <= 1):
        print("Error: --threshold must be between 0 and 1")
        sys.exit(1)
    
    # Generate output path
    if args.output:
        output_path = Path(args.output)
        # Add .jpg extension if no extension provided
        if not output_path.suffix:
            output_path = output_path.with_suffix('.jpg')
            print(f"No extension specified, using: {output_path}")
    else:
        # Auto-generate output path in results folder with UUID
        if args.debug:
            results_dir = Path("results/debug")
        else:
            results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        # Generate 4-character UUID prefix
        uuid_prefix = uuid.uuid4().hex[:4]
        # Include threshold in filename
        threshold_str = f"t{args.threshold}".replace(".", "_")
        output_filename = f"{video_path.stem}_{uuid_prefix}_{threshold_str}.jpg"
        output_path = results_dir / output_filename
        print(f"No output specified, using: {output_path}")
    
    try:
        if args.debug:
            # Debug mode: analyze changes only
            print("Debug mode: Analyzing changes and generating threshold graph")
            
            if args.xcolumn is not None:
                print(f"Column mode: Analyzing vertical line at x={args.xcolumn}")
                analyze_changes_only(video_path, x_column=args.xcolumn, debug_output=output_path)
            else:
                print(f"Row mode: Analyzing horizontal line at y={args.yrow}")
                analyze_changes_only(video_path, y_row=args.yrow, debug_output=output_path)
            
            print("Change analysis completed successfully!")
        else:
            # Normal mode: extract strip photography
            if args.xcolumn is not None:
                print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
                extract_column_strip(video_path, args.xcolumn, output_path, args.threshold, args.relax)
            else:
                print(f"Row mode: Extracting horizontal line at y={args.yrow}")
                extract_row_strip(video_path, args.yrow, output_path, args.threshold, args.relax)
            
            print("Strip photography extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()