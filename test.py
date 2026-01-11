import os
import cv2

# Replace this with your path
base_dir = r"C:\Users\Babureddy B\OneDrive\Documents\Desktop\ROAD_N\data"

# Directories to create
directories = ['train', 'valid', 'test']

# Check OpenCV installation
try:
    img = cv2.imread(os.path.join(base_dir, 'train', 'accident', 'acc1 (2).jpg'))
    if img is not None:
        print("✓ OpenCV is working correctly")
    else:
        print("⚠ Could not read test image")
except Exception as e:
    print(f"⚠ Error with OpenCV: {str(e)}")

# Create and verify directories
for dir_name in directories:
    labels_dir = os.path.join(base_dir, dir_name, 'labels')
    try:
        os.makedirs(labels_dir, exist_ok=True)
        print(f"✓ Created/verified {labels_dir}")
    except Exception as e:
        print(f"⚠ Error creating {labels_dir}: {str(e)}")

print("\nSetup check complete!")