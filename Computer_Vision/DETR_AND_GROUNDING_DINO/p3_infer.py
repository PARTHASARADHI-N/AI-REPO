import os
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run YOLOv8 inference on validation dataset and save predictions to JSON.")
parser.add_argument("val_images_root", type=str, help="Path to the validation images root folder")
parser.add_argument("model_path", type=str, help="Path to the fine-tuned YOLOv8 model (e.g., best.pt)")
parser.add_argument("output_json", type=str, help="Path and filename for the output predictions JSON file")
args = parser.parse_args()


VAL_IMAGES_ROOT = args.val_images_root
MODEL_PATH = args.model_path
OUTPUT_JSON = args.output_json
CONF_THRESHOLD = 0.25 
IOU_THRESHOLD = 0.5    

if not os.path.exists(VAL_IMAGES_ROOT):
    print(f"Error: Validation images root folder {VAL_IMAGES_ROOT} does not exist.")
    exit(1)
if not os.path.isfile(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found.")
    exit(1)


OUTPUT_DIR = os.path.dirname(OUTPUT_JSON)
if OUTPUT_DIR:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


print(f"Loading fine-tuned model from {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

class_names = model.names  
print(f"Model class names: {class_names}")


coco_predictions = {
    "images": [],
    "annotations": []
}
annotation_id = 1
image_id = 1


image_extensions = ('.jpg', '.jpeg', '.png')
image_files = sorted([f for f in os.listdir(VAL_IMAGES_ROOT) if f.lower().endswith(image_extensions)])
if not image_files:
    print(f"Error: No images found in {VAL_IMAGES_ROOT} with extensions {image_extensions}.")
    exit(1)


print("\nRunning inference on validation set...")
for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(VAL_IMAGES_ROOT, img_name)
    if not os.path.isfile(img_path):
        print(f"Warning: Image {img_path} not found, skipping.")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}, skipping.")
        continue
    
    
    coco_predictions["images"].append({
        "id": image_id,
        "file_name": img_name
    })
    

    try:
        results = model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        pred_result = results[0]
    except Exception as e:
        print(f"Warning: Inference failed for {img_path}: {str(e)}")
        continue
    

    for box in pred_result.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box[:6]
        class_id = int(cls)
        
        if class_id in class_names:
            width = round(x2 - x1, 1)
            height = round(y2 - y1, 1)
            x1, y1 = round(x1, 1), round(y1, 1)
            
            coco_predictions["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,  
                "bbox": [x1, y1, width, height],
                "score": round(float(conf), 3) 
            })
            
            annotation_id += 1
    
    image_id += 1


print(f"\nTotal images processed: {len(coco_predictions['images'])}")
print(f"Total predictions: {len(coco_predictions['annotations'])}")
try:
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_predictions, f, indent=2)
    print(f"Saved predictions to {OUTPUT_JSON}")
except Exception as e:
    print(f"Error saving predictions to {OUTPUT_JSON}: {str(e)}")
    exit(1)

print("\n Inference complete.")