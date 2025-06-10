
import os
import argparse
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from transformers import (
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection
)

class LoadData(Dataset):
    def __init__(self, annotation_file, image_root):
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_root = image_root

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.coco.imgs[image_id]
        path = os.path.join(self.image_root, info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        annotations = [
            {
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0)
            }
            for ann in anns
        ]
        return image, {"image_id": image_id, "annotations": annotations}

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def find_annotation_file(image_dir):
 
    json_files = glob.glob(os.path.join(image_dir, "*.json"))
    
    if not json_files:
        parent_dir = os.path.dirname(image_dir)
        json_files = glob.glob(os.path.join(parent_dir, "*.json"))
        
        ann_dir = os.path.join(parent_dir, "annotations")
        if os.path.exists(ann_dir):
            json_files.extend(glob.glob(os.path.join(ann_dir, "*.json")))
   
    ann_files = [f for f in json_files if "ann" in os.path.basename(f).lower()]
  
    if ann_files:
        return ann_files[0]
    elif json_files:
        return json_files[0]
    
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=int, choices=[1, 2, 3], help='Training mode: 1=encoder, 2=decoder, 3=all')
    parser.add_argument('train_image_root', type=str, help='Path to training images directory')
    parser.add_argument('output', type=str, help='Full path to save model checkpoint (e.g., ./checkpoints/model.pth)')
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help="List of score thresholds for mAP evaluation (unused)")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file (optional, will auto-detect if not provided)")
    return parser.parse_args()

def freeze_parameters(model, mode):
    if mode == 1:
        for name, param in model.named_parameters(): 
            param.requires_grad = ("encoder" in name)
    elif mode == 2:
        for name, param in model.named_parameters(): 
            param.requires_grad = ("decoder" in name)
    else:
        for param in model.parameters(): 
            param.requires_grad = True

def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    freeze_parameters(model, args.mode)
    model.to(device)


    annotation_file = args.annotation_file
    if annotation_file is None:
        annotation_file = find_annotation_file(args.train_image_root)
        if annotation_file is None:
            raise FileNotFoundError("Could not find a suitable annotation file. Please provide one with --annotation_file")
        print(f"Using detected annotation file: {annotation_file}")

    train_ds = LoadData(
        annotation_file=annotation_file,
        image_root=args.train_image_root
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=4
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_dl, start=1):
            raw = processor(images=images, annotations=targets, return_tensors="pt")
            enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
            if "labels" in raw:
                enc["labels"] = [
                    {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in ann.items()}
                    for ann in raw["labels"]
                ]

            outputs = model(**enc)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch[{epoch}/{args.epochs}] Step[{step}/{len(train_dl)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_dl)
        epoch_losses.append(avg_loss)
        print(f"→ Finished Epoch {epoch} | Avg Loss: {avg_loss:.4f}")


        torch.save(model.state_dict(), args.output)
        print(f"Saved model checkpoint: {args.output}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, 'o-', linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Avg Loss", fontsize=14)
    plt.title("Training Loss", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot = os.path.splitext(args.output)[0] + "_loss.png"
    plt.savefig(loss_plot, dpi=300)
    plt.close()
    print(f"Saved loss plot: {loss_plot}")

if __name__ == "__main__":
    main()