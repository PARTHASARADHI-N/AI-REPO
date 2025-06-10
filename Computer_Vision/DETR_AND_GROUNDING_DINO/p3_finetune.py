import os
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
import yaml
import shutil
import argparse

parser = argparse.ArgumentParser(description="Train YOLO model with custom dataset")
parser.add_argument('--input_folder', type=str, required=True, help="Path to input folder containing images and COCO JSON annotation file")
parser.add_argument('--output_path', type=str, required=True, help="Full path to save the model (e.g., /path/to/output/model.pt)")
args = parser.parse_args()

TRAIN_IMAGES_ROOT = args.input_folder  
OUTPUT_DIR = os.path.dirname(args.output_path)  
MODEL_SAVE_PATH = args.output_path  
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "train/labels")
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640

TRAIN_ANNOTATION_FILE = None
for file in os.listdir(args.input_folder):
    if file.endswith('.json'):
        TRAIN_ANNOTATION_FILE = os.path.join(args.input_folder, file)
        break

if TRAIN_ANNOTATION_FILE is None:
    print("Error: No JSON annotation file found in input folder.")
    exit(1)

print(f"Input folder: {args.input_folder}")
print(f"Output model path: {MODEL_SAVE_PATH}")
print(f"Annotation file: {TRAIN_ANNOTATION_FILE}")


os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True) 


print(f"Loading model: yolov8n.pt")
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit(1)

print(f"Loading training annotations from {TRAIN_ANNOTATION_FILE}")
try:
    coco_train = COCO(TRAIN_ANNOTATION_FILE)
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_train.loadCats(coco_train.getCatIds())}
    class_names = list(category_id_to_name.values())
    

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
 
    print("Class mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
except FileNotFoundError:
    print(f"Error: {TRAIN_ANNOTATION_FILE} not found.")
    exit(1)

def convert_coco_to_yolo(coco, images_root, images_output_dir, labels_output_dir, name_to_idx):
    img_count = 0
    label_count = 0
    
    for img in tqdm(coco.loadImgs(coco.getImgIds()), desc="Converting annotations"):
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        

        src_img_path = os.path.join(images_root, img_name)
        dst_img_path = os.path.join(images_output_dir, os.path.basename(img_name))
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
            img_count += 1
        else:
            print(f"Warning: Image {src_img_path} not found, skipping.")
            continue
        

        label_filename = os.path.splitext(os.path.basename(img_name))[0] + '.txt'
        label_path = os.path.join(labels_output_dir, label_filename)
        with open(label_path, 'w') as f:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            for ann in anns:
                x, y, w, h = ann['bbox']
                category_name = category_id_to_name[ann['category_id']]
                if category_name in name_to_idx:
                    class_id = name_to_idx[category_name]
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    label_count += 1
    print(f"Copied {img_count} images and generated {label_count} label entries.")
    return img_count, label_count


print("Converting training annotations to YOLO format...")
train_img_count, train_label_count = convert_coco_to_yolo(coco_train, TRAIN_IMAGES_ROOT, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, name_to_idx)


print("\nVerifying dataset structure...")
train_images = len([f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
train_labels = len([f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')])
print(f"Training: {train_images} images, {train_labels} labels")


print("\nSample label content:")
sample_files = [f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')][:3]
for sample in sample_files:
    with open(os.path.join(TRAIN_LABELS_DIR, sample), 'r') as f:
        content = f.read()
    print(f"File: {sample}\nContent: {content[:100]}...")


data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
data_yaml = {
    "train": TRAIN_IMAGES_DIR,
    "val": TRAIN_IMAGES_DIR,  
    "nc": len(class_names),
    "names": class_names
}
with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f)
print(f"Created data.yaml at {data_yaml_path}")


print("\ndata.yaml contents:")
with open(data_yaml_path, "r") as f:
    print(f.read())

print("\nStarting fine-tuning...")
try:
    model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=OUTPUT_DIR,
        name="yolov8n_finetuned",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        momentum=0.937,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=20,
        save=True,
        device=0,
        cache=False
    )
except Exception as e:
    print(f"Error during training: {str(e)}")
    exit(1)
print("Fine-tuning complete.")


best_model_temp_path = os.path.join(OUTPUT_DIR, "yolov8n_finetuned", "weights", "best.pt")
try:
    if os.path.exists(best_model_temp_path):
        shutil.move(best_model_temp_path, MODEL_SAVE_PATH)
        print(f"Model saved at: {MODEL_SAVE_PATH}")
    else:
        print(f"Error: Best model not found at {best_model_temp_path}. Training may have failed.")
        exit(1)
except Exception as e:
    print(f"Error saving model to {MODEL_SAVE_PATH}: {str(e)}")
    exit(1)


print("\n Training complete.")
print(f" Model saved at: {MODEL_SAVE_PATH}")
print(f" Dataset prepared at: {DATASET_DIR}")
print(f" Data Quality Checks:")
print(f" Training images/labels ratio: {train_images}/{train_labels}")