#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os
from pathlib import Path

def apply_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=15):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_edge_detection(image):
    """Apply Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Canny(gray, 50, 150)

def apply_histogram_equalization(image):
    """Apply histogram equalization"""
    if len(image.shape) == 3:
        # For color images, convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # For grayscale images
        return cv2.equalizeHist(image)

def apply_adaptive_threshold(image):
    """Apply adaptive thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def apply_morphological_operations(image):
    """Apply morphological closing operation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

def apply_bilateral_filter(image):
    """Apply bilateral filter for noise reduction while preserving edges"""
    return cv2.bilateralFilter(image, 9, 75, 75)

def apply_sharpening(image):
    """Apply sharpening filter"""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def apply_contrast_brightness(image, alpha=1.5, beta=30):
    """Adjust contrast and brightness"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'.")
        sys.exit(1)
    
    # Create output directory
    base_name = Path(image_path).stem
    output_dir = f"processed_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply preprocessing techniques and save results
    techniques = [
        ("original", lambda img: img),
        ("grayscale", apply_grayscale),
        ("gaussian_blur", apply_gaussian_blur),
        ("edge_detection", apply_edge_detection),
        ("histogram_equalization", apply_histogram_equalization),
        ("adaptive_threshold", apply_adaptive_threshold),
        ("morphological_closing", apply_morphological_operations),
        ("bilateral_filter", apply_bilateral_filter),
        ("sharpening", apply_sharpening),
        ("contrast_brightness", apply_contrast_brightness)
    ]
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    print("Applying preprocessing techniques...")
    
    for technique_name, technique_func in techniques:
        try:
            processed_image = technique_func(image.copy())
            
            # Convert single channel images to 3-channel for consistent saving
            if len(processed_image.shape) == 2:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            
            output_path = os.path.join(output_dir, f"{base_name}_{technique_name}.jpg")
            cv2.imwrite(output_path, processed_image)
            print(f" {technique_name}: {output_path}")
            
        except Exception as e:
            print(f" Error applying {technique_name}: {str(e)}")
    
    print(f"\nProcessing complete! Check the '{output_dir}' directory for results.")

if __name__ == "__main__":
    main()