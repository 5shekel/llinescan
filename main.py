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
from pathlib import Path


def extract_column_strip(video_path, x_column, output_path):
    """
    Extract vertical strip at x_column from each frame of the video.
    
    Args:
        video_path: Path to input video file
        x_column: X-coordinate of the column to extract
        output_path: Path for output image
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
    
    # Initialize output array: (height, total_frames, 3)
    strip_image = np.zeros((frame_height, total_frames, 3), dtype=np.uint8)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < total_frames:
            # Extract column at x_column and store in strip_image
            strip_image[:, frame_idx, :] = frame[:, x_column, :]
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    print(f"Output dimensions: {strip_image.shape}")
    print(f"Saving to: {output_path}")
    
    # Save the strip image
    cv2.imwrite(str(output_path), strip_image)


def extract_row_strip(video_path, y_row, output_path):
    """
    Extract horizontal strip at y_row from each frame of the video.
    
    Args:
        video_path: Path to input video file
        y_row: Y-coordinate of the row to extract
        output_path: Path for output image
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
    
    # Initialize output array: (total_frames, width, 3)
    strip_image = np.zeros((total_frames, frame_width, 3), dtype=np.uint8)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < total_frames:
            # Extract row at y_row and store in strip_image
            strip_image[frame_idx, :, :] = frame[y_row, :, :]
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
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
    
    output_path = Path(args.output)
    
    try:
        if args.xcolumn is not None:
            print(f"Column mode: Extracting vertical line at x={args.xcolumn}")
            extract_column_strip(video_path, args.xcolumn, output_path)
        else:
            print(f"Row mode: Extracting horizontal line at y={args.yrow}")
            extract_row_strip(video_path, args.yrow, output_path)
        
        print("Strip photography extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()