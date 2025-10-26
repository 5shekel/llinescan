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
        # Generate change graph
        graph_path = f"{debug_output}_changes.png"
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
            print(f"\nSuggested threshold values:")
            for p in percentiles:
                thresh = np.percentile(changes, p)
                frames_above = np.sum(np.array(changes) >= thresh)
                compression = (len(changes) - frames_above) / len(changes) * 100
                print(f"  {p}th percentile: {thresh:.4f} (keeps {frames_above} frames, {compression:.1f}% compression)")
    
    return changes


def extract_column_strip(video_path, x_column, output_path, change_threshold=0.01):
    """
    Extract vertical strip at x_column from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
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
    
    # Collect significant columns
    significant_columns = []
    previous_column = None
    included_frames = 0
    skipped_frames = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract current column
        current_column = frame[:, x_column, :].copy()
        
        # Check if this is the first frame or if change is significant
        include_frame = False
        if previous_column is None:
            include_frame = True  # Always include first frame
        else:
            change = calculate_line_difference(current_column, previous_column)
            if change >= change_threshold:
                include_frame = True
        
        if include_frame:
            significant_columns.append(current_column)
            previous_column = current_column
            included_frames += 1
        else:
            skipped_frames += 1
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
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


def extract_row_strip(video_path, y_row, output_path, change_threshold=0.01):
    """
    Extract horizontal strip at y_row from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
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
    
    # Collect significant rows
    significant_rows = []
    previous_row = None
    included_frames = 0
    skipped_frames = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract current row
        current_row = frame[y_row, :, :].copy()
        
        # Check if this is the first frame or if change is significant
        include_frame = False
        if previous_row is None:
            include_frame = True  # Always include first frame
        else:
            change = calculate_line_difference(current_row, previous_row)
            if change >= change_threshold:
                include_frame = True
        
        if include_frame:
            significant_rows.append(current_row)
            previous_row = current_row
            included_frames += 1
        else:
            skipped_frames += 1
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    if not significant_rows:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    # Convert list to numpy array
    strip_image = np.stack(significant_rows, axis=0)
    
    print(f"Original frames: {total_frames}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/total_frames:.1%}")
    print(f"Output dimensions: {strip_image.shape}")
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
        help="Extract horizontal line at y-coordinate (row mode)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output image file path"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Change threshold (0-1) for including frames (default: 0.01)"
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
    
    if args.xcolumn is None and args.yrow is None:
        print("Error: Must specify either --xcolumn or --yrow.")
        sys.exit(1)
    
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
    
    output_path = Path(args.output)
    
    try:
        if args.debug:
            # Debug mode: analyze changes only
            print("Debug mode: Analyzing changes and generating threshold graph")
            
            if args.xcolumn is not None:
                print(f"Column mode: Analyzing vertical line at x={args.xcolumn}")
                analyze_changes_only(video_path, x_column=args.xcolumn, debug_output=output_path.stem)
            else:
                print(f"Row mode: Analyzing horizontal line at y={args.yrow}")
                analyze_changes_only(video_path, y_row=args.yrow, debug_output=output_path.stem)
            
            print("Change analysis completed successfully!")
        else:
            # Normal mode: extract strip photography
            if args.xcolumn is not None:
                print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
                extract_column_strip(video_path, args.xcolumn, output_path, args.threshold)
            else:
                print(f"Row mode: Extracting horizontal line at y={args.yrow}")
                extract_row_strip(video_path, args.yrow, output_path, args.threshold)
            
            print("Strip photography extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()