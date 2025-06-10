import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


parser = argparse.ArgumentParser(description='Object detection with Grounding DINO')
parser.add_argument('image_dir', type=str, help='Path to input image directory')
parser.add_argument('output_path', type=str, help='Path and filename for output prediction JSON')
args = parser.parse_args()


OUTPUT_DIR = os.path.dirname(args.output_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


model_name = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(DEVICE)
model.eval()


CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "train"},
    {"id": 4, "name": "rider"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "motorcycle"},
    {"id": 7, "name": "bicycle"},
    {"id": 8, "name": "bus"}
]


CATEGORY_TO_PROMPT = {
    "person": "a person",
    "car": "a car",
    "train": "a train",
    "rider": "a rider",
    "truck": "a truck",
    "motorcycle": "a motorcycle",
    "bicycle": "a bicycle",
    "bus": "a bus"
}

class SimpleImageDataset(Dataset):
    """Simple dataset that loads images from a directory"""
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()  
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
       
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        return image, {"image_id": idx + 1, "file_name": img_name}

def custom_collate_fn(batch):
    images, metadata = zip(*batch)
    return list(images), list(metadata)

dataset = SimpleImageDataset(args.image_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)


def post_process_outputs(outputs, target_sizes, threshold=0.005):
    logits = outputs["logits"].sigmoid()
    boxes = outputs["pred_boxes"]

    boxes = boxes[0]
    logits = logits[0]
    scores = logits[:, 0]

    mask = scores > threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    img_h, img_w = target_sizes[0]
    filtered_boxes[:, 0] = filtered_boxes[:, 0] * img_w
    filtered_boxes[:, 1] = filtered_boxes[:, 1] * img_h
    filtered_boxes[:, 2] = filtered_boxes[:, 2] * img_w
    filtered_boxes[:, 3] = filtered_boxes[:, 3] * img_h

    filtered_boxes[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    filtered_boxes[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    filtered_boxes[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2]
    filtered_boxes[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3]

    return {"scores": filtered_scores, "boxes": filtered_boxes}


annotations = []
images = []
detection_id = 1

print(f"Starting inference on {len(dataloader)} images for {len(CATEGORIES)} categories")


for i, (img, metadata) in enumerate(tqdm(dataloader, desc="Processing images")):
    pil_img = img[0]
    width, height = pil_img.size
    target_size = torch.tensor([[height, width]])
    img_id = metadata[0]["image_id"]
    file_name = metadata[0]["file_name"]
    
    images.append({
        "id": img_id,
        "file_name": file_name
    })
    

    for category in CATEGORIES:
        category_name = category["name"]
        text_prompt = CATEGORY_TO_PROMPT[category_name]
        category_id = category["id"]
        
        inputs = processor(images=pil_img, text=[text_prompt], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        result = post_process_outputs(outputs, target_size, threshold=0.005)
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        
        high_confidence_indices = scores >= 0.005
        filtered_boxes = boxes[high_confidence_indices]
        filtered_scores = scores[high_confidence_indices]

        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            

            if x2 <= x1 or y2 <= y1:
                continue
                
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            annotations.append({
                "id": detection_id,
                "image_id": int(img_id),
                "category_id": int(category_id),
                "bbox": [round(b, 2) for b in coco_box],
                "score": float(round(score, 3))
            })
            detection_id += 1
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(dataloader)} images.")


output_data = {
    "images": images,
    "annotations": annotations
}


with open(args.output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"All predictions saved to: {args.output_path}")
print(f"Total predictions: {len(annotations)}")
print(f"Total images: {len(images)}")


predictions_per_category = {}
for cat in CATEGORIES:
    cat_id = cat["id"]
    cat_name = cat["name"]
    count = len([p for p in annotations if p["category_id"] == cat_id])
    predictions_per_category[cat_name] = count
    print(f"Category '{cat_name}': {count} predictions")

print("Detection complete!")