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
import math


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


def analyze_changes_only(video_path, x_column=None, y_row=None, debug_output=None, start_frame=0, end_frame=None):
    """
    Analyze changes in video without generating strip image.
    Used for debug mode to generate change threshold graphs.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of column to analyze (if column mode)
        y_row: Y-coordinate of row to analyze (if row mode)
        debug_output: Base path for debug outputs
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
        
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames) for change analysis...")
    
    changes = []
    previous_line = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
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
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Analyzed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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


def add_timeline_overlay(image, frame_numbers):
    """
    Add frame number overlay as a timeline/ruler at the bottom of the image.
    Always horizontal from left to right.
    
    Args:
        image: The strip image to add overlay to
        frame_numbers: List of frame numbers that were included
    
    Returns:
        Image with timeline overlay
    """
    if not frame_numbers:
        return image
    
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_color = (255, 255, 0)  # Cyan for visibility
    
    # Calculate text size for spacing
    (text_width, text_height), _ = cv2.getTextSize("00000", font, font_scale, font_thickness)
    
    # Horizontal timeline at the bottom from left to right
    # Calculate spacing to avoid overlap
    available_width = image.shape[1]
    image_height = image.shape[0]
    num_labels = min(len(frame_numbers), max(10, available_width // (text_width + 10)))
    step = max(1, len(frame_numbers) // num_labels)
    
    for i in range(0, len(frame_numbers), step):
        frame_num = frame_numbers[i]
        text = str(frame_num)
        x_pos = int((i / len(frame_numbers)) * available_width)
        
        # Add small tick mark at bottom
        cv2.line(overlay, (x_pos, image_height - 10), (x_pos, image_height), text_color, 1)
        # Add text above tick mark
        cv2.putText(overlay, text, (x_pos + 2, image_height - 12),
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return overlay


def extract_column_strip(video_path, x_column, output_path, change_threshold=0.005, relax=0, timeline=False, start_frame=0, end_frame=None):
    """
    Extract vertical strip at x_column from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        timeline: If True, overlay frame numbers as timeline
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting column {x_column} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all columns and identify significant frames
    all_columns = []
    changes = []
    frame_numbers = []
    previous_column = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current column
        current_column = frame[:, x_column, :].copy()
        all_columns.append(current_column)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_column is not None:
            change = calculate_line_difference(current_column, previous_column)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_column = current_column
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    
    # Collect significant columns with actual frame numbers
    significant_columns = []
    significant_frame_numbers = []
    for i, col in enumerate(all_columns):
        if include_mask[i]:
            significant_columns.append(col)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_columns) - included_frames
    
    if not significant_columns:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    # Convert list to numpy array
    strip_image = np.stack(significant_columns, axis=1)
    
    # Flip horizontally so time flows from right to left (strip photography convention)
    strip_image = cv2.flip(strip_image, 1)
    
    # Add timeline overlay if requested
    if timeline:
        strip_image = add_timeline_overlay(strip_image, significant_frame_numbers)
    
    print(f"Original frames in segment: {len(all_columns)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/total_frames:.1%}")
    print(f"Output dimensions: {strip_image.shape}")
    print(f"Saving to: {output_path}")
    
    # Save the strip image
    cv2.imwrite(str(output_path), strip_image)


def extract_row_strip(video_path, y_row, output_path, change_threshold=0.01, relax=0, timeline=False, start_frame=0, end_frame=None):
    """
    Extract horizontal strip at y_row from each frame of the video.
    Only include frames where the change exceeds the threshold.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output image
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        timeline: If True, overlay frame numbers as timeline
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting row {y_row} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all rows and identify significant frames
    all_rows = []
    changes = []
    frame_numbers = []
    previous_row = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current row
        current_row = frame[y_row, :, :].copy()
        all_rows.append(current_row)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_row is not None:
            change = calculate_line_difference(current_row, previous_row)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_row = current_row
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    
    # Collect significant rows with actual frame numbers
    significant_rows = []
    significant_frame_numbers = []
    for i, row in enumerate(all_rows):
        if include_mask[i]:
            significant_rows.append(row)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_rows) - included_frames
    
    if not significant_rows:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    # Convert list to numpy array
    strip_image = np.stack(significant_rows, axis=0)
    
    # Rotate clockwise 90 degrees for row mode
    strip_image = cv2.rotate(strip_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Flip horizontally so time flows from right to left (strip photography convention)
    strip_image = cv2.flip(strip_image, 1)
    
    # Add timeline overlay if requested (after rotation and flip)
    if timeline:
        strip_image = add_timeline_overlay(strip_image, significant_frame_numbers)
    
    print(f"Original frames in segment: {len(all_rows)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/total_frames:.1%}")
    print(f"Output dimensions: {strip_image.shape} (rotated 90° CW)")
    print(f"Saving to: {output_path}")
    
    # Save the strip image
    cv2.imwrite(str(output_path), strip_image)


def add_timestamp_overlay(frame, frame_count, total_frames):
    """
    Add frame count overlay to the bottom left corner of the frame.
    
    Args:
        frame: The video frame to add overlay to
        frame_count: Current frame number (1-based)
        total_frames: Total number of frames
        
    Returns:
        Frame with timestamp overlay
    """
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (0, 255, 255)  # Yellow for visibility
    bg_color = (0, 0, 0)  # Black background
    
    # Create timestamp text
    timestamp_text = f"Frame: {frame_count}/{total_frames}"
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)
    
    # Position at bottom left with some padding
    x_pos = 10
    y_pos = frame.shape[0] - 10  # Bottom of frame minus padding
    
    # Draw background rectangle
    cv2.rectangle(overlay,
                 (x_pos - 5, y_pos - text_height - baseline - 5),
                 (x_pos + text_width + 5, y_pos + baseline + 5),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(overlay, timestamp_text, (x_pos, y_pos - baseline),
               font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return overlay


def extract_column_strip_video(video_path, x_column, output_path, change_threshold=0.005, relax=0, start_frame=0, end_frame=None, fps=30, timestamp=False):
    """
    Extract vertical strip at x_column from each frame and create an MJPEG video.
    Each frame of the output video shows the accumulated scan lines up to that point.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output video file
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
        fps: Output video frame rate
        timestamp: If True, embed frame count on bottom left corner
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting column {x_column} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all columns and identify significant frames
    all_columns = []
    changes = []
    frame_numbers = []
    previous_column = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current column
        current_column = frame[:, x_column, :].copy()
        all_columns.append(current_column)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_column is not None:
            change = calculate_line_difference(current_column, previous_column)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_column = current_column
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    significant_columns = []
    significant_frame_numbers = []
    for i, col in enumerate(all_columns):
        if include_mask[i]:
            significant_columns.append(col)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_columns) - included_frames
    
    if not significant_columns:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    print(f"Original frames in segment: {len(all_columns)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/len(all_columns):.1%}")
    
    # Create video writer
    # Output video dimensions: height = input frame height, width = number of significant frames (final)
    final_output_width = len(significant_columns)
    final_output_height = frame_height
    
    print(f"Output video dimensions: {final_output_width}x{final_output_height}")
    print(f"Creating MJPEG video at {fps} FPS: {output_path}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (final_output_width, final_output_height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create video writer for: {output_path}")
    
    # Generate video frames - each frame shows accumulated scan lines up to that point
    for frame_idx in range(len(significant_columns)):
        # Create accumulated strip image up to current frame
        accumulated_columns = significant_columns[:frame_idx + 1]
        
        # Convert to numpy array and create the frame
        strip_frame = np.stack(accumulated_columns, axis=1)
        
        # Flip horizontally so time flows from right to left (strip photography convention)
        strip_frame = cv2.flip(strip_frame, 1)
        
        # Pad the frame to match the final video dimensions
        current_height, current_width = strip_frame.shape[:2]
        if current_width < final_output_width or current_height < final_output_height:
            # Create a black frame of the final size
            padded_frame = np.zeros((final_output_height, final_output_width, 3), dtype=strip_frame.dtype)
            # Copy the current frame to the right side (for progressive width growth from right to left)
            padded_frame[:current_height, final_output_width-current_width:] = strip_frame
            strip_frame = padded_frame
        
        # Add timestamp overlay if requested (after padding)
        if timestamp:
            strip_frame = add_timestamp_overlay(strip_frame, frame_idx + 1, len(significant_columns))
        
        # Write frame to video
        out.write(strip_frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Generated {frame_idx + 1}/{len(significant_columns)} video frames")
    
    # Release video writer
    out.release()
    
    print(f"MJPEG video saved to: {output_path}")
    print(f"Video contains {len(significant_columns)} frames at {fps} FPS")
    print(f"Total duration: {len(significant_columns)/fps:.2f} seconds")


def extract_row_strip_video(video_path, y_row, output_path, change_threshold=0.01, relax=0, start_frame=0, end_frame=None, fps=30, timestamp=False):
    """
    Extract horizontal strip at y_row from each frame and create an MJPEG video.
    Each frame of the output video shows the accumulated scan lines up to that point.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output video file
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
        fps: Output video frame rate
        timestamp: If True, embed frame count on bottom left corner
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting row {y_row} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all rows and identify significant frames
    all_rows = []
    changes = []
    frame_numbers = []
    previous_row = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current row
        current_row = frame[y_row, :, :].copy()
        all_rows.append(current_row)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_row is not None:
            change = calculate_line_difference(current_row, previous_row)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_row = current_row
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    significant_rows = []
    significant_frame_numbers = []
    for i, row in enumerate(all_rows):
        if include_mask[i]:
            significant_rows.append(row)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_rows) - included_frames
    
    if not significant_rows:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    print(f"Original frames in segment: {len(all_rows)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/len(all_rows):.1%}")
    
    # Create video writer
    # For row mode, we rotate CCW 90°: output video dimensions after rotation
    # Before rotation: height = frame_idx + 1 (progressive), width = input frame width
    # After rotation: height = input frame width, width = frame_idx + 1 (progressive)
    # We'll set dimensions to the final size for the video container
    final_output_width = len(significant_rows)  # After rotation
    final_output_height = frame_width  # After rotation
    
    print(f"Output video dimensions (after rotation): {final_output_width}x{final_output_height}")
    print(f"Creating MJPEG video at {fps} FPS: {output_path}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (final_output_width, final_output_height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create video writer for: {output_path}")
    
    # Generate video frames - each frame shows accumulated scan lines up to that point
    for frame_idx in range(len(significant_rows)):
        # Create accumulated strip image up to current frame
        accumulated_rows = significant_rows[:frame_idx + 1]
        
        # Convert to numpy array and create the frame
        strip_frame = np.stack(accumulated_rows, axis=0)
        
        # Rotate counter-clockwise 90 degrees to match image mode orientation
        strip_frame = cv2.rotate(strip_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Flip horizontally so time flows from right to left (strip photography convention)
        strip_frame = cv2.flip(strip_frame, 1)
        
        # Pad the frame to match the final video dimensions
        current_height, current_width = strip_frame.shape[:2]
        if current_width < final_output_width or current_height < final_output_height:
            # Create a black frame of the final size
            padded_frame = np.zeros((final_output_height, final_output_width, 3), dtype=strip_frame.dtype)
            # Copy the current frame to the right side (for progressive width growth from right to left)
            padded_frame[:current_height, final_output_width-current_width:] = strip_frame
            strip_frame = padded_frame
        
        # Add timestamp overlay if requested (after padding)
        if timestamp:
            strip_frame = add_timestamp_overlay(strip_frame, frame_idx + 1, len(significant_rows))
        
        # Write frame to video
        out.write(strip_frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Generated {frame_idx + 1}/{len(significant_rows)} video frames")
    
    # Release video writer
    out.release()
    
    print(f"MJPEG video saved to: {output_path}")
    print(f"Video contains {len(significant_rows)} frames at {fps} FPS")
    print(f"Total duration: {len(significant_rows)/fps:.2f} seconds")


def extract_column_strip_alpha(video_path, x_column, output_path, change_threshold=0.005, relax=0, start_frame=0, end_frame=None, fps=30, timestamp=False):
    """
    Extract vertical strip at x_column from each frame and create PNG sequence with alpha transparency.
    Each frame shows the accumulated scan lines up to that point with transparent background.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output directory (PNG sequence)
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
        fps: Output video frame rate (for reference)
        timestamp: If True, embed frame count on bottom left corner
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting column {x_column} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all columns and identify significant frames
    all_columns = []
    changes = []
    frame_numbers = []
    previous_column = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current column
        current_column = frame[:, x_column, :].copy()
        all_columns.append(current_column)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_column is not None:
            change = calculate_line_difference(current_column, previous_column)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_column = current_column
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    significant_columns = []
    significant_frame_numbers = []
    for i, col in enumerate(all_columns):
        if include_mask[i]:
            significant_columns.append(col)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_columns) - included_frames
    
    if not significant_columns:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    print(f"Original frames in segment: {len(all_columns)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/len(all_columns):.1%}")
    
    # Create output directory
    # For column mode: width = number of significant frames, height = input frame height
    final_output_width = len(significant_columns)
    final_output_height = frame_height
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output PNG sequence dimensions: {final_output_width}x{final_output_height}")
    print(f"Creating PNG sequence at {fps} FPS reference: {output_dir}")
    
    # Generate PNG frames - each frame shows accumulated scan lines up to that point
    for frame_idx in range(len(significant_columns)):
        # Create accumulated strip image up to current frame
        accumulated_columns = significant_columns[:frame_idx + 1]
        
        # Convert to numpy array and create the frame with alpha channel
        strip_frame_bgr = np.stack(accumulated_columns, axis=1)
        
        # Flip horizontally so time flows from right to left (strip photography convention)
        strip_frame_bgr = cv2.flip(strip_frame_bgr, 1)
        
        # Create BGRA frame with alpha channel
        current_height, current_width = strip_frame_bgr.shape[:2]
        strip_frame_bgra = np.zeros((final_output_height, final_output_width, 4), dtype=np.uint8)
        
        # Copy RGB data to BGR channels and set alpha to 255 for actual content
        # Place content on the right side for progressive growth from right to left
        strip_frame_bgra[:current_height, final_output_width-current_width:, :3] = strip_frame_bgr
        strip_frame_bgra[:current_height, final_output_width-current_width:, 3] = 255  # Opaque for content
        # Transparent areas remain alpha=0
        
        # Add timestamp overlay if requested (after alpha setup)
        if timestamp:
            # Convert back to BGR for timestamp overlay, then back to BGRA
            bgr_for_timestamp = strip_frame_bgra[:, :, :3].copy()
            bgr_with_timestamp = add_timestamp_overlay(bgr_for_timestamp, frame_idx + 1, len(significant_columns))
            strip_frame_bgra[:, :, :3] = bgr_with_timestamp
        
        # Save PNG frame with zero-padded frame number
        frame_filename = f"frame_{frame_idx:06d}.png"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), strip_frame_bgra)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Generated {frame_idx + 1}/{len(significant_columns)} PNG frames")
    
    print(f"PNG sequence saved to: {output_dir}")
    print(f"Sequence contains {len(significant_columns)} frames at {fps} FPS reference")
    print(f"Total duration: {len(significant_columns)/fps:.2f} seconds")
    print(f"Import into video editor as PNG sequence at {fps} FPS")


def extract_row_strip_alpha(video_path, y_row, output_path, change_threshold=0.01, relax=0, start_frame=0, end_frame=None, fps=30, timestamp=False):
    """
    Extract horizontal strip at y_row from each frame and create PNG sequence with alpha transparency.
    Each frame shows the accumulated scan lines up to that point with transparent background.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output directory (PNG sequence)
        change_threshold: Minimum change threshold (0-1) to include frame
        relax: Number of extra frames to include before/after threshold frames
        start_frame: First frame to process (0-based)
        end_frame: Last frame to process (None = until end)
        fps: Output video frame rate (for reference)
        timestamp: If True, embed frame count on bottom left corner
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
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames - 1
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    print(f"Extracting row {y_row} from {frame_width}x{frame_height} frames")
    print(f"Change threshold: {change_threshold}")
    if relax > 0:
        print(f"Relax: including {relax} frames before/after threshold frames")
    
    # First pass: collect all rows and identify significant frames
    all_rows = []
    changes = []
    frame_numbers = []
    previous_row = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames before start
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        
        # Stop after end frame
        if frame_idx > end_frame:
            break
            
        # Extract current row
        current_row = frame[y_row, :, :].copy()
        all_rows.append(current_row)
        frame_numbers.append(frame_idx)
        
        # Calculate change from previous frame
        if previous_row is not None:
            change = calculate_line_difference(current_row, previous_row)
            changes.append(change)
        else:
            changes.append(0)  # First frame has no change
        
        previous_row = current_row
        frame_idx += 1
        
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processed {frame_idx - start_frame}/{end_frame - start_frame + 1} frames")
    
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
    significant_rows = []
    significant_frame_numbers = []
    for i, row in enumerate(all_rows):
        if include_mask[i]:
            significant_rows.append(row)
            significant_frame_numbers.append(frame_numbers[i])
    
    included_frames = sum(include_mask)
    skipped_frames = len(all_rows) - included_frames
    
    if not significant_rows:
        raise ValueError("No significant changes detected. Try lowering the threshold.")
    
    print(f"Original frames in segment: {len(all_rows)}")
    print(f"Included frames: {included_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Compression ratio: {skipped_frames/len(all_rows):.1%}")
    
    # Create output directory
    final_output_width = len(significant_rows)  # After rotation
    final_output_height = frame_width  # After rotation
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output PNG sequence dimensions (after rotation): {final_output_width}x{final_output_height}")
    print(f"Creating PNG sequence at {fps} FPS reference: {output_dir}")
    
    # Generate PNG frames - each frame shows accumulated scan lines up to that point
    for frame_idx in range(len(significant_rows)):
        # Create accumulated strip image up to current frame
        accumulated_rows = significant_rows[:frame_idx + 1]
        
        # Convert to numpy array and create the frame
        strip_frame_bgr = np.stack(accumulated_rows, axis=0)
        
        # Rotate counter-clockwise 90 degrees to match image mode orientation
        strip_frame_bgr = cv2.rotate(strip_frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Flip horizontally so time flows from right to left (strip photography convention)
        strip_frame_bgr = cv2.flip(strip_frame_bgr, 1)
        
        # Create BGRA frame with alpha channel
        current_height, current_width = strip_frame_bgr.shape[:2]
        strip_frame_bgra = np.zeros((final_output_height, final_output_width, 4), dtype=np.uint8)
        
        # Copy RGB data to BGR channels and set alpha to 255 for actual content
        # Place content on the right side for progressive growth from right to left
        strip_frame_bgra[:current_height, final_output_width-current_width:, :3] = strip_frame_bgr
        strip_frame_bgra[:current_height, final_output_width-current_width:, 3] = 255  # Opaque for content
        # Transparent areas remain alpha=0
        
        # Add timestamp overlay if requested (after alpha setup)
        if timestamp:
            # Convert back to BGR for timestamp overlay, then back to BGRA
            bgr_for_timestamp = strip_frame_bgra[:, :, :3].copy()
            bgr_with_timestamp = add_timestamp_overlay(bgr_for_timestamp, frame_idx + 1, len(significant_rows))
            strip_frame_bgra[:, :, :3] = bgr_with_timestamp
        
        # Save PNG frame with zero-padded frame number
        frame_filename = f"frame_{frame_idx:06d}.png"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), strip_frame_bgra)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Generated {frame_idx + 1}/{len(significant_rows)} PNG frames")
    
    print(f"PNG sequence saved to: {output_dir}")
    print(f"Sequence contains {len(significant_rows)} frames at {fps} FPS reference")
    print(f"Total duration: {len(significant_rows)/fps:.2f} seconds")
    print(f"Import into video editor as PNG sequence at {fps} FPS")


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
        help="Output file path (default: results/<input_name>.jpg for images, .avi for videos)"
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
        "--start",
        type=int,
        default=0,
        help="Start frame number (0-based, default: 0)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        help="End frame number (0-based, default: last frame)"
    )
    
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Overlay frame numbers as timeline/ruler on output image (image mode only)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: analyze changes and generate threshold graph without creating strip image"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate MJPEG video showing accumulated scan lines over time"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate (default: 30.0, only used with --video)"
    )
    
    parser.add_argument(
        "--timestamp",
        "--ts",
        action="store_true",
        help="Embed frame count on bottom left corner (video mode only)"
    )
    
    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Generate PNG sequence with alpha transparency for video editing (video mode only)"
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
    
    # Validate frame range
    if args.start < 0:
        print("Error: --start must be non-negative")
        sys.exit(1)
    
    if args.end is not None and args.end < args.start:
        print("Error: --end must be greater than or equal to --start")
        sys.exit(1)
    
    # Validate video mode arguments
    if args.video and args.timeline:
        print("Warning: --timeline is not supported in video mode, ignoring")
        args.timeline = False
    
    if args.video and args.debug:
        print("Error: Cannot use --video and --debug modes together")
        sys.exit(1)
    
    if args.alpha and not args.video:
        print("Error: --alpha can only be used with --video mode")
        sys.exit(1)
    
    # Validate FPS
    if args.fps <= 0:
        print("Error: --fps must be positive")
        sys.exit(1)
    
    # Generate output path
    if args.output:
        output_path = Path(args.output)
        # Add appropriate extension if no extension provided
        if not output_path.suffix:
            if args.video and args.alpha:
                # For alpha mode, we'll create a directory for PNG sequence
                output_path = output_path.with_suffix('')  # Remove any extension
                print(f"No extension specified for alpha video mode, using directory: {output_path}")
            elif args.video:
                output_path = output_path.with_suffix('.avi')
                print(f"No extension specified for video mode, using: {output_path}")
            else:
                output_path = output_path.with_suffix('.jpg')
                print(f"No extension specified for image mode, using: {output_path}")
    else:
        # Auto-generate output path in results folder with UUID
        if args.debug:
            results_dir = Path("results/debug")
        elif args.video:
            results_dir = Path("results/video")
        else:
            results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        # Generate 4-character UUID prefix
        uuid_prefix = uuid.uuid4().hex[:4]
        # Include threshold in filename
        threshold_str = f"t{args.threshold}".replace(".", "_")
        
        if args.video and args.alpha:
            fps_str = f"fps{args.fps}".replace(".", "_")
            output_filename = f"{video_path.stem}_{uuid_prefix}_{threshold_str}_{fps_str}_alpha"
        elif args.video:
            fps_str = f"fps{args.fps}".replace(".", "_")
            output_filename = f"{video_path.stem}_{uuid_prefix}_{threshold_str}_{fps_str}.avi"
        else:
            output_filename = f"{video_path.stem}_{uuid_prefix}_{threshold_str}.jpg"
        
        output_path = results_dir / output_filename
        print(f"No output specified, using: {output_path}")
    
    try:
        if args.debug:
            # Debug mode: analyze changes only
            print("Debug mode: Analyzing changes and generating threshold graph")
            
            if args.xcolumn is not None:
                print(f"Column mode: Analyzing vertical line at x={args.xcolumn}")
                analyze_changes_only(video_path, x_column=args.xcolumn, debug_output=output_path,
                                   start_frame=args.start, end_frame=args.end)
            else:
                print(f"Row mode: Analyzing horizontal line at y={args.yrow}")
                analyze_changes_only(video_path, y_row=args.yrow, debug_output=output_path,
                                   start_frame=args.start, end_frame=args.end)
            
            print("Change analysis completed successfully!")
        elif args.video and args.alpha:
            # Alpha video mode: create PNG sequence with alpha transparency
            print("Alpha video mode: Creating PNG sequence with alpha transparency")
            
            if args.xcolumn is not None:
                print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
                extract_column_strip_alpha(video_path, args.xcolumn, output_path, args.threshold, args.relax,
                                         args.start, args.end, args.fps, args.timestamp)
            else:
                print(f"Row mode: Extracting horizontal line at y={args.yrow}")
                extract_row_strip_alpha(video_path, args.yrow, output_path, args.threshold, args.relax,
                                      args.start, args.end, args.fps, args.timestamp)
            
            print("Alpha PNG sequence generation completed successfully!")
        elif args.video:
            # Video mode: create MJPEG video with accumulated scan lines
            print("Video mode: Creating MJPEG video with accumulated scan lines")
            
            if args.xcolumn is not None:
                print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
                extract_column_strip_video(video_path, args.xcolumn, output_path, args.threshold, args.relax,
                                         args.start, args.end, args.fps, args.timestamp)
            else:
                print(f"Row mode: Extracting horizontal line at y={args.yrow}")
                extract_row_strip_video(video_path, args.yrow, output_path, args.threshold, args.relax,
                                      args.start, args.end, args.fps, args.timestamp)
            
            print("MJPEG video generation completed successfully!")
        else:
            # Normal mode: extract strip photography image
            print("Image mode: Creating strip photography image")
            
            if args.xcolumn is not None:
                print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
                extract_column_strip(video_path, args.xcolumn, output_path, args.threshold, args.relax, args.timeline,
                                   args.start, args.end)
            else:
                print(f"Row mode: Extracting horizontal line at y={args.yrow}")
                extract_row_strip(video_path, args.yrow, output_path, args.threshold, args.relax, args.timeline,
                                args.start, args.end)
            
            print("Strip photography extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()