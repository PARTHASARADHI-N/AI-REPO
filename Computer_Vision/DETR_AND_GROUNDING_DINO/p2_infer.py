# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, ToTensor
# from torchvision.datasets import CocoDetection
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from tqdm import tqdm
# import json
# from PIL import Image, ImageDraw
# import numpy as np
# from torchvision.ops import nms
# import traceback


# val_root = "/kaggle/input/val-foggy/foggy_dataset_A3_val"
# val_ann = "/kaggle/input/ann-val-train/annotations_val.json"
# pt_file_path = "task3_model.pt"
# output_dir = "output_predictions"
# batch_size = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# score_threshold = 0.5  
# nms_iou_threshold = 0.2  
# max_boxes_per_image = 50 
# max_boxes_per_class = 10 
# num_prompt_tokens = 4  


# os.makedirs(output_dir, exist_ok=True)

# def load_category_names(ann_file):
#     """Load category names from COCO annotation file."""
#     with open(ann_file, 'r') as f:
#         data = json.load(f)
#     return {cat['id']: cat['name'] for cat in data['categories']}

# category_map = load_category_names(val_ann)
# class_names = list(category_map.values())  

# class CustomCocoDetection(CocoDetection):
#     def _load_image(self, id: int) -> Image.Image:
#         path = self.coco.loadImgs(id)[0]["file_name"]
#         full_path = os.path.join(self.root, path)
#         return Image.open(full_path).convert("RGB")

#     def __getitem__(self, idx):
#         try:
#             img_id = self.ids[idx]
#             image = self._load_image(img_id)
#             target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
#             return image, target, img_id
#         except Exception as e:
#             print(f"Error loading image at index {idx}: {e}")
#             return Image.new('RGB', (800, 800)), [], None

# try:
#     val_ds = CustomCocoDetection(root=val_root, annFile=val_ann, transform=None)
# except Exception as e:
#     raise ValueError(f"Error loading dataset: {e}")


# def collate_fn(batch):
#     images, targets, img_ids = [], [], []
#     for item in batch:
#         images.append(item[0])
#         targets.append(item[1])
#         img_ids.append(item[2])
#     return images, targets, img_ids

# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
# model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
# model.eval()


# text_embedding_layer = None
# for name, module in model.named_modules():
#     if isinstance(module, nn.Embedding) and "word_embeddings" in name.lower():
#         text_embedding_layer = module
#         break

# if text_embedding_layer is None:
#     raise ValueError("Text embedding layer not found in model")


# virtual_prompt = torch.load(pt_file_path, map_location=device, weights_only=True)
# num_virtual_tokens, embed_dim = virtual_prompt.shape


# if num_virtual_tokens > num_prompt_tokens:
#     virtual_prompt = virtual_prompt[:num_prompt_tokens]
#     num_virtual_tokens = num_prompt_tokens
#     print(f"Truncated virtual prompts to {num_virtual_tokens} tokens")


# def hook_fn(module, input, output):
#     if virt_embeds is not None:
#         if output.shape[1] >= num_virtual_tokens:
#             output[:, :num_virtual_tokens] = virt_embeds
#             print(f"Applied hook for {num_virtual_tokens} tokens, output shape: {output.shape}")
#         else:
#             print(f"Warning: Output sequence length {output.shape[1]} is less than num_virtual_tokens {num_virtual_tokens}")
#     return output

# hook = text_embedding_layer.register_forward_hook(hook_fn)


# def process_predictions(outputs, original_sizes, processor_size, class_name):
#     processed = []
#     for i, (logits, boxes) in enumerate(zip(outputs.logits, outputs.pred_boxes)):
#         orig_w, orig_h = original_sizes[i]
#         proc_h, proc_w = processor_size
        

#         boxes = boxes * torch.tensor([proc_w, proc_h, proc_w, proc_h], device=device)
#         boxes = center_to_corners(boxes)
        

#         boxes[:, [0, 2]] *= (orig_w / proc_w)
#         boxes[:, [1, 3]] *= (orig_h / proc_h)

#         scores = torch.softmax(logits, -1).max(-1).values
#         keep = scores > score_threshold
#         boxes, scores = boxes[keep], scores[keep]
#         labels = [class_name] * len(boxes)  
        

#         if len(boxes) > 0:
#             print(f"Class {class_name}: {len(boxes)} boxes before NMS, scores: {scores.cpu().numpy().round(2)}")
#             keep = nms(boxes, scores, nms_iou_threshold)
#             boxes = boxes[keep]
#             scores = scores[keep]
#             labels = [labels[k] for k in keep]
#             print(f"Class {class_name}: {len(boxes)} boxes after NMS")
            

#             if len(boxes) > max_boxes_per_class:
#                 indices = torch.argsort(scores, descending=True)[:max_boxes_per_class]
#                 boxes = boxes[indices]
#                 scores = scores[indices]
#                 labels = [labels[i] for i in indices]
        
#         processed.append({
#             "boxes": boxes.cpu().numpy(),
#             "scores": scores.cpu().numpy(),
#             "labels": labels
#         })
#     return processed

# def center_to_corners(boxes):
#     """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
#     return torch.stack([
#         boxes[:, 0] - boxes[:, 2] / 2,
#         boxes[:, 1] - boxes[:, 3] / 2,
#         boxes[:, 0] + boxes[:, 2] / 2,
#         boxes[:, 1] + boxes[:, 3] / 2,
#     ], dim=1)


# def draw_boxes(image, predictions, output_path, class_name):
#     draw = ImageDraw.Draw(image)
#     colors = {
#         "car": "red",
#         "bus": "blue",
#         "truck": "green",
#         "person": "yellow",
#         "bicycle": "purple",
#         "motorcycle": "orange",
#         "traffic light": "cyan",
#         "stop sign": "pink"
#     }
#     color = colors.get(class_name, "red") if class_name != "combined" else "red"
#     for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
#         draw.rectangle(box.tolist(), outline=color, width=2)
#         draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill=color)
#     image.save(output_path)


# def apply_cross_class_nms(predictions):
#     if not predictions["boxes"].size:
#         return predictions
#     boxes = torch.tensor(predictions["boxes"], device=device)
#     scores = torch.tensor(predictions["scores"], device=device)
#     keep = nms(boxes, scores, nms_iou_threshold)
#     return {
#         "boxes": boxes[keep].cpu().numpy(),
#         "scores": scores[keep].cpu().numpy(),
#         "labels": [predictions["labels"][k] for k in keep]
#     }


# image_counter = 0
# print("Starting inference...")
# with torch.no_grad():
#     for batch_idx, (images, targets, img_ids) in enumerate(tqdm(val_loader, desc="Inference")):
#         try:

#             original_sizes = [(img.width, img.height) for img in images]
#             processor_size = None
 
#             all_predictions = [[] for _ in range(len(images))]
            

#             for class_name in class_names:

#                 inputs = processor(
#                     text=[f"{class_name} ."] * len(images),
#                     images=images,
#                     return_tensors="pt",
#                     padding=True
#                 ).to(device)
                

#                 print(f"Class {class_name}: Input tokens: {inputs['input_ids'].shape}, {inputs['input_ids']}")

#                 if processor_size is None:
#                     processor_size = inputs["pixel_values"].shape[-2:]

#                 global virt_embeds
#                 virt_embeds = virtual_prompt.unsqueeze(0).expand(len(images), -1, -1)

#                 outputs = model(**inputs)

#                 predictions = process_predictions(outputs, original_sizes, processor_size, class_name)

#                 for i in range(len(images)):
#                     if img_ids[i] is None:
#                         continue
#                     file_name = val_ds.coco.loadImgs(img_ids[i])[0]["file_name"]
#                     class_output_dir = os.path.join(output_dir, class_name)
#                     output_path = os.path.join(class_output_dir, file_name)
#                     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#                     draw_boxes(images[i].copy(), predictions[i], output_path, class_name)
#                     all_predictions[i].append(predictions[i])
            

#             for i in range(len(images)):
#                 combined_predictions = {
#                     "boxes": [],
#                     "scores": [],
#                     "labels": []
#                 }
#                 for pred in all_predictions[i]:
#                     if len(pred["boxes"]) > 0:
#                         combined_predictions["boxes"].extend(pred["boxes"])
#                         combined_predictions["scores"].extend(pred["scores"])
#                         combined_predictions["labels"].extend(pred["labels"])

#                 if combined_predictions["boxes"]:
#                     combined_predictions["boxes"] = np.array(combined_predictions["boxes"])
#                     combined_predictions["scores"] = np.array(combined_predictions["scores"])
                    

#                     print(f"Image {img_ids[i]}: {len(combined_predictions['boxes'])} boxes before cross-class NMS")
#                     combined_predictions = apply_cross_class_nms(combined_predictions)
#                     print(f"Image {img_ids[i]}: {len(combined_predictions['boxes'])} boxes after cross-class NMS")
                    

#                     if len(combined_predictions["boxes"]) > max_boxes_per_image:
#                         indices = np.argsort(combined_predictions["scores"])[-max_boxes_per_image:]
#                         combined_predictions["boxes"] = combined_predictions["boxes"][indices]
#                         combined_predictions["scores"] = combined_predictions["scores"][indices]
#                         combined_predictions["labels"] = [combined_predictions["labels"][i] for i in indices]

#                 if img_ids[i] is None:
#                     continue
#                 file_name = val_ds.coco.loadImgs(img_ids[i])[0]["file_name"]
#                 combined_output_dir = os.path.join(output_dir, "combined")
#                 output_path = os.path.join(combined_output_dir, file_name)
#                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
#                 draw_boxes(images[i].copy(), combined_predictions, output_path, "combined")
            
#             image_counter += len(images)
#         except Exception as e:
#             print(f"Error in batch {batch_idx}: {str(e)}")
#             print(f"Input shapes: {{k: v.shape for k, v in inputs.items() if isinstance(v, torch.Tensor)}}")
#             traceback.print_exc()
#             continue

# # Cleanup
# hook.remove()
# print(f"Processed {image_counter} images")
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CocoDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm
import json
from PIL import Image, ImageDraw
import numpy as np
from torchvision.ops import nms
import traceback
import argparse


parser = argparse.ArgumentParser(description="Run inference with Grounding DINO model")
parser.add_argument('input_dir', type=str, help="Path to input directory containing images and COCO JSON annotation file")
parser.add_argument('output_file', type=str, help="Full path to save the predictions JSON file (e.g., /path/to/output/predictions.json)")
args = parser.parse_args()


val_root = args.input_dir
val_ann = None
for file in os.listdir(args.input_dir):
    if file.endswith('.json'):
        val_ann = os.path.join(args.input_dir, file)
        break

if val_ann is None:
    raise ValueError("Error: No JSON annotation file found in input directory.")

pt_file_path = "task2_model.pt"
output_dir = "output_predictions"
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_threshold = 0.5  
nms_iou_threshold = 0.2  
max_boxes_per_image = 50 
max_boxes_per_class = 10 
num_prompt_tokens = 4  

os.makedirs(output_dir, exist_ok=True)

def load_category_names(ann_file):
    """Load category names from COCO annotation file."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return {cat['id']: cat['name'] for cat in data['categories']}

category_map = load_category_names(val_ann)
class_names = list(category_map.values())  

class CustomCocoDetection(CocoDetection):
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        full_path = os.path.join(self.root, path)
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx):
        try:
            img_id = self.ids[idx]
            image = self._load_image(img_id)
            target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            return image, target, img_id
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return Image.new('RGB', (800, 800)), [], None

try:
    val_ds = CustomCocoDetection(root=val_root, annFile=val_ann, transform=None)
except Exception as e:
    raise ValueError(f"Error loading dataset: {e}")

def collate_fn(batch):
    images, targets, img_ids = [], [], []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
        img_ids.append(item[2])
    return images, targets, img_ids

val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
model.eval()

text_embedding_layer = None
for name, module in model.named_modules():
    if isinstance(module, nn.Embedding) and "word_embeddings" in name.lower():
        text_embedding_layer = module
        break

if text_embedding_layer is None:
    raise ValueError("Text embedding layer not found in model")

virtual_prompt = torch.load(pt_file_path, map_location=device, weights_only=True)
num_virtual_tokens, embed_dim = virtual_prompt.shape

if num_virtual_tokens > num_prompt_tokens:
    virtual_prompt = virtual_prompt[:num_prompt_tokens]
    num_virtual_tokens = num_prompt_tokens
    print(f"Truncated virtual prompts to {num_virtual_tokens} tokens")

def hook_fn(module, input, output):
    if virt_embeds is not None:
        if output.shape[1] >= num_virtual_tokens:
            output[:, :num_virtual_tokens] = virt_embeds
            print(f"Applied hook for {num_virtual_tokens} tokens, output shape: {output.shape}")
        else:
            print(f"Warning: Output sequence length {output.shape[1]} is less than num_virtual_tokens {num_virtual_tokens}")
    return output

hook = text_embedding_layer.register_forward_hook(hook_fn)

def process_predictions(outputs, original_sizes, processor_size, class_name):
    processed = []
    for i, (logits, boxes) in enumerate(zip(outputs.logits, outputs.pred_boxes)):
        orig_w, orig_h = original_sizes[i]
        proc_h, proc_w = processor_size
        
        boxes = boxes * torch.tensor([proc_w, proc_h, proc_w, proc_h], device=device)
        boxes = center_to_corners(boxes)
        
        boxes[:, [0, 2]] *= (orig_w / proc_w)
        boxes[:, [1, 3]] *= (orig_h / proc_h)

        scores = torch.softmax(logits, -1).max(-1).values
        keep = scores > score_threshold
        boxes, scores = boxes[keep], scores[keep]
        labels = [class_name] * len(boxes)  
        
        if len(boxes) > 0:
            print(f"Class {class_name}: {len(boxes)} boxes before NMS, scores: {scores.cpu().numpy().round(2)}")
            keep = nms(boxes, scores, nms_iou_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[k] for k in keep]
            print(f"Class {class_name}: {len(boxes)} boxes after NMS")
            
            if len(boxes) > max_boxes_per_class:
                indices = torch.argsort(scores, descending=True)[:max_boxes_per_class]
                boxes = boxes[indices]
                scores = scores[indices]
                labels = [labels[i] for i in indices]
        
        processed.append({
            "boxes": boxes.cpu().numpy(),
            "scores": boxes.cpu().numpy(),
            "labels": labels
        })
    return processed

def center_to_corners(boxes):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    return torch.stack([
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2,
    ], dim=1)

def draw_boxes(image, predictions, output_path, class_name):
    draw = ImageDraw.Draw(image)
    colors = {
        "car": "red",
        "bus": "blue",
        "truck": "green",
        "person": "yellow",
        "bicycle": "purple",
        "motorcycle": "orange",
        "traffic light": "cyan",
        "stop sign": "pink"
    }
    color = colors.get(class_name, "red") if class_name != "combined" else "red"
    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        draw.rectangle(box.tolist(), outline=color, width=2)
        draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill=color)
    image.save(output_path)

def apply_cross_class_nms(predictions):
    if not predictions["boxes"].size:
        return predictions
    boxes = torch.tensor(predictions["boxes"], device=device)
    scores = torch.tensor(predictions["scores"], device=device)
    keep = nms(boxes, scores, nms_iou_threshold)
    return {
        "boxes": boxes[keep].cpu().numpy(),
        "scores": scores[keep].cpu().numpy(),
        "labels": [predictions["labels"][k] for k in keep]
    }


coco_predictions = {
    "images": [],
    "annotations": []
}
annotation_id = 1
name_to_category_id = {name: id for id, name in category_map.items()}

image_counter = 0
print("Starting inference...")
with torch.no_grad():
    for batch_idx, (images, targets, img_ids) in enumerate(tqdm(val_loader, desc="Inference")):
        try:
            original_sizes = [(img.width, img.height) for img in images]
            processor_size = None
 
            all_predictions = [[] for _ in range(len(images))]
            
            for class_name in class_names:
                inputs = processor(
                    text=[f"{class_name} ."] * len(images),
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                print(f"Class {class_name}: Input tokens: {inputs['input_ids'].shape}, {inputs['input_ids']}")

                if processor_size is None:
                    processor_size = inputs["pixel_values"].shape[-2:]

                global virt_embeds
                virt_embeds = virtual_prompt.unsqueeze(0).expand(len(images), -1, -1)

                outputs = model(**inputs)

                predictions = process_predictions(outputs, original_sizes, processor_size, class_name)

                for i in range(len(images)):
                    if img_ids[i] is None:
                        continue
                    file_name = val_ds.coco.loadImgs(img_ids[i])[0]["file_name"]
                    class_output_dir = os.path.join(output_dir, class_name)
                    output_path = os.path.join(class_output_dir, file_name)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    draw_boxes(images[i].copy(), predictions[i], output_path, class_name)
                    all_predictions[i].append(predictions[i])
            
            for i in range(len(images)):
                if img_ids[i] is None:
                    continue
                # Add image to coco_predictions
                coco_predictions["images"].append({
                    "id": img_ids[i],
                    "file_name": val_ds.coco.loadImgs(img_ids[i])[0]["file_name"]
                })
                
                combined_predictions = {
                    "boxes": [],
                    "scores": [],
                    "labels": []
                }
                for pred in all_predictions[i]:
                    if len(pred["boxes"]) > 0:
                        combined_predictions["boxes"].extend(pred["boxes"])
                        combined_predictions["scores"].extend(pred["scores"])
                        combined_predictions["labels"].extend(pred["labels"])

                if combined_predictions["boxes"]:
                    combined_predictions["boxes"] = np.array(combined_predictions["boxes"])
                    combined_predictions["scores"] = np.array(combined_predictions["scores"])
                    
                    print(f"Image {img_ids[i]}: {len(combined_predictions['boxes'])} boxes before cross-class NMS")
                    combined_predictions = apply_cross_class_nms(combined_predictions)
                    print(f"Image {img_ids[i]}: {len(combined_predictions['boxes'])} boxes after cross-class NMS")
                    
                    if len(combined_predictions["boxes"]) > max_boxes_per_image:
                        indices = np.argsort(combined_predictions["scores"])[-max_boxes_per_image:]
                        combined_predictions["boxes"] = combined_predictions["boxes"][indices]
                        combined_predictions["scores"] = combined_predictions["scores"][indices]
                        combined_predictions["labels"] = [combined_predictions["labels"][i] for i in indices]

                    # Add annotations to coco_predictions
                    for box, score, label in zip(combined_predictions["boxes"], combined_predictions["scores"], combined_predictions["labels"]):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        coco_predictions["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_ids[i],
                            "category_id": name_to_category_id[label],
                            "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(width), 1), round(float(height), 1)],
                            "score": round(float(score), 4)
                        })
                        annotation_id += 1

                file_name = val_ds.coco.loadImgs(img_ids[i])[0]["file_name"]
                combined_output_dir = os.path.join(output_dir, "combined")
                output_path = os.path.join(combined_output_dir, file_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                draw_boxes(images[i].copy(), combined_predictions, output_path, "combined")
            
            image_counter += len(images)
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Input shapes: {{k: v.shape for k, v in inputs.items() if isinstance(v, torch.Tensor)}}")
            traceback.print_exc()
            continue

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "w") as f:
    json.dump(coco_predictions, f, indent=2)
print(f"Saved COCO predictions to {args.output_file}")

hook.remove()
print(f"Processed {image_counter} images")