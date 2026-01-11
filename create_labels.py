import os
import glob
from pathlib import Path
import cv2

def create_label_file(image_path):
    """
    Create a YOLO format label file for an image
    Class 0: accident
    Class 1: no_accident
    """
    # Get image dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    # Determine class based on filename
    is_accident = 'accident' in Path(image_path).name.lower() and 'no_accident' not in Path(image_path).name.lower()
    class_id = 0 if is_accident else 1
    
    # Create label path - same name as image but in labels directory
    image_dir = Path(image_path).parent
    label_dir = os.path.join(str(Path(image_path).parent.parent), 'labels')
    os.makedirs(label_dir, exist_ok=True)
    
    label_path = os.path.join(label_dir, Path(image_path).stem + '.txt')
    
    # Write the label file (center coordinates and full image size)
    with open(label_path, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    return label_path

def process_dataset(data_dir):
    """Process all images in the dataset and create corresponding label files"""
    # Define directories
    directories = ['train', 'valid', 'test']
    
    total_processed = 0
    total_accident = 0
    total_no_accident = 0
    
    for dir_name in directories:
        img_dir = os.path.join(data_dir, dir_name, 'images')
        if not os.path.exists(img_dir):
            print(f"Directory not found: {img_dir}")
            continue
        
        # Process all images in directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
        print(f"\nProcessing {dir_name} directory:")
        for img_path in image_files:
            label_path = create_label_file(img_path)
            if label_path:
                total_processed += 1
                if 'accident' in Path(img_path).name.lower() and 'no_accident' not in Path(img_path).name.lower():
                    total_accident += 1
                else:
                    total_no_accident += 1
                print(f"Created label for: {Path(img_path).name}")
    
    print(f"\nProcessing complete!")
    print(f"Total labels created: {total_processed}")
    print(f"Accident images labeled: {total_accident}")
    print(f"No-accident images labeled: {total_no_accident}")

if __name__ == "__main__":
    # Update this with your dataset directory path
    dataset_dir = r"C:\Users\Babureddy B\OneDrive\Documents\Desktop\ROAD_N\data"
    
    print("Starting label creation process...")
    process_dataset(dataset_dir)
    print("Label creation completed!")