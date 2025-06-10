
import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
from transformers import (
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor
)


parser = argparse.ArgumentParser(description="Run inference and optional evaluation with Deformable DETR.")
parser.add_argument('input_dir', type=str, help='Path to the input image directory')
parser.add_argument('model_path', type=str, help='Path to the .pt model file')
parser.add_argument('output', type=str, help='Full path to save the predictions JSON (e.g., output/preds.json)')
parser.add_argument('--val_annotation_file', type=str, default=None, help='Path to the COCO-style validation annotation file (optional)')
args = parser.parse_args()

output_dir = os.path.dirname(args.output)
os.makedirs(output_dir, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")


model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
model.eval()
model.to(DEVICE) 
processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

dataset_to_coco_mapping = {
    1: 1,   # person
    2: 7,   # car
    3: 2,   # train
    4: 6,   # rider
    6: 8,   # motorcycle
    7: 3,   # bicycle
    8: 5    # bus
}

class LoadData(Dataset):
    def __init__(self, image_root, annotation_file=None):
        self.image_root = image_root
        if annotation_file:
            self.coco = COCO(annotation_file)
            self.image_ids = list(self.coco.imgs.keys())
            self.has_annotations = True
        else:
            self.image_paths = [
                os.path.join(dp, f)
                for dp, _, filenames in os.walk(image_root)
                for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.image_ids = list(range(len(self.image_paths)))
            self.has_annotations = False

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.has_annotations:
            img_info = self.coco.loadImgs(image_id)[0]
            path = os.path.join(self.image_root, img_info["file_name"])
            image = Image.open(path).convert("RGB")
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
            boxes = [ann["bbox"] for ann in anns]
            labels = [ann["category_id"] for ann in anns]
            return image, {"boxes": boxes, "labels": labels, "image_id": image_id, "file_name": img_info["file_name"]}
        else:
            path = self.image_paths[idx]
            image = Image.open(path).convert("RGB")
            fname = os.path.relpath(path, self.image_root)
            return image, {"image_id": image_id, "file_name": fname}


def custom_collate_fn(batch):
    return list(zip(*batch))

dataset = LoadData(args.input_dir, args.val_annotation_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

results = []
for i, (imgs, metas) in enumerate(dataloader):
    img = imgs[0]
    meta = metas[0]
    width, height = img.size
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

   
    processed = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([[height, width]]), threshold=0.3)[0]
    boxes = processed["boxes"].cpu().numpy()
    scores = processed["scores"].cpu().numpy()
    labels = processed["labels"].cpu().numpy()

    results.append({
        "image_id": meta["image_id"],
        "file_name": meta["file_name"],
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    })


prediction_json = {"images": [], "annotations": []}
added_ids = set()
ann_id = 1

for result in results:
    img_id = result["image_id"]
    file_name = result["file_name"]
    if img_id not in added_ids:
        prediction_json["images"].append({"id": img_id, "file_name": file_name})
        added_ids.add(img_id)

    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        coco_label = dataset_to_coco_mapping.get(label.item(), int(label.item()))
        prediction_json["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(coco_label),
            "bbox": [round(float(x_min), 2), round(float(y_min), 2), round(float(width), 2), round(float(height), 2)],
            "score": round(float(score), 3)

        })
        ann_id += 1


with open(args.output, 'w') as f:
    json.dump(prediction_json, f)
print(f"Predictions saved to: {args.output}")


if args.val_annotation_file:
    coco_gt = COCO(args.val_annotation_file)
    coco_dt = coco_gt.loadRes(args.output)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP50:95": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
    }
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
