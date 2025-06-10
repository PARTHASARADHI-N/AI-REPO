import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CocoDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, BertTokenizer
from tqdm import tqdm
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train soft prompts for zero-shot object detection using Grounding DINO")
    parser.add_argument('input_root', type=str,
                        help="Root directory containing the training image folder and annotation JSON file")
    parser.add_argument('output_path', type=str,
                        help="Output file path (including filename) to save the learned soft prompts (.pt)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for optimizer")
    return parser.parse_args()


def find_annotation_file(root_dir):
    """Search for the first .json file under the root directory and return its path."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No JSON annotation file found under {root_dir}")


if __name__ == '__main__':
    args = parse_args()


    train_root = args.input_root
    train_ann = find_annotation_file(args.input_root)

    batch_size = args.batch_size
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_normalized_coords = True
    use_center_format = True
    min_box_size = 0.002

    def load_category_names(ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        return {cat['id']: cat['name'] for cat in data['categories']}

    def validate_annotations(ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        for ann in data['annotations']:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                pass
        for img in data['images']:
            if img['width'] <= 0 or img['height'] <= 0:
                pass

    category_map = load_category_names(train_ann)
    validate_annotations(train_ann)


    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    for p in model.parameters():
        p.requires_grad = False

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text_embedding_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "word_embeddings" in name.lower() and module.weight.shape[0] > 1000:
            text_embedding_layer = module
            break
    if text_embedding_layer is None:
        raise ValueError("Could not find word embedding layer in model.")
    embed_dim = text_embedding_layer.embedding_dim


    class PromptLearner(nn.Module):
        def __init__(self, num_virtual_tokens, embed_dim):
            super().__init__()
            self.virtual_prompt = nn.Parameter(torch.randn(num_virtual_tokens, embed_dim))
        def forward(self, batch_size):
            return self.virtual_prompt.unsqueeze(0).expand(batch_size, -1, -1)

    num_virtual_tokens = 4
    prompt_learner = PromptLearner(num_virtual_tokens, embed_dim).to(device)


    virt_embeds_tensor = None
    def embedding_hook(module, input, output):
        global virt_embeds_tensor
        if virt_embeds_tensor is not None and output.shape == virt_embeds_tensor.shape:
            return virt_embeds_tensor.clone()
        return output
    hook_handle = text_embedding_layer.register_forward_hook(embedding_hook)

 
    def debug_hook(name):
        def hook(module, input, output):
            pass
        return hook
    hook_handles = []
    for name, module in model.named_modules():
        if any(key in name.lower() for key in ('text_backbone', 'bbox_head', 'loss')):
            hook_handles.append(module.register_forward_hook(debug_hook(name)))


    transform = Compose([ToTensor()])
    def preprocess_targets(targets, image_sizes, processed_image_sizes):
        processed = []
        for target, (orig_h, orig_w), (proc_h, proc_w) in zip(targets, image_sizes, processed_image_sizes):
            boxes, labels, class_names = [], [], []
            for ann in target:
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0: continue
                x2, y2 = x + w, y + h
                x, y, x2, y2 = x*proc_w/orig_w, y*proc_h/orig_h, x2*proc_w/orig_w, y2*proc_h/orig_h
                if use_center_format:
                    box = [(x+x2)/2, (y+y2)/2, x2-x, y2-y]
                else:
                    box = [x, y, x2, y2]
                if use_normalized_coords:
                    if use_center_format:
                        box = [c/proc_w if i%2==0 else c/proc_h for i,c in enumerate(box)]
                    else:
                        box = [box[0]/proc_w, box[1]/proc_h, box[2]/proc_w, box[3]/proc_h]
                if box[2] < min_box_size or box[3] < min_box_size: continue
                boxes.append(box); labels.append(ann['category_id']); class_names.append(category_map.get(ann['category_id'], 'unknown'))
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device) if boxes else torch.empty(0,4).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device) if labels else torch.empty(0,dtype=torch.int64).to(device)
            processed.append({'boxes': boxes, 'labels': labels, 'class_labels': class_names})
        return processed

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        image_sizes = [(img.shape[1], img.shape[2]) for img in images]
        processed = processor(images=images, text=['object']*len(images), return_tensors='pt')
        proc_sizes = [(processed['pixel_values'][i].shape[1],
                       processed['pixel_values'][i].shape[2]) for i in range(len(images))]
        targets = preprocess_targets(targets, image_sizes, proc_sizes)
        return images, targets

    train_ds = CocoDetection(root=train_root, annFile=train_ann, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=args.lr)

    for epoch in range(num_epochs):
        prompt_learner.train()
        running_loss, valid_batches = 0.0, 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            bs = images.size(0)
            virt_embeds_tensor = prompt_learner(bs)
            inputs = processor(images=images, text=['object']*bs, return_tensors='pt')
            inputs = {k: v.to(device) for k,v in inputs.items()}
            
            labels = [{
                'boxes': t['boxes'], 'class_labels': t['class_labels']
            } for t in preprocess_targets(targets, [(img.shape[1], img.shape[2]) for img in images],
                [(inputs['pixel_values'][i].shape[1], inputs['pixel_values'][i].shape[2]) for i in range(bs)]) if t['boxes'].nelement() > 0]
            if not labels: continue
            inputs['labels'] = labels
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is None: continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item(); valid_batches += 1
        avg_loss = running_loss / max(valid_batches,1)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

    hook_handle.remove()
    for handle in hook_handles: handle.remove()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(prompt_learner.virtual_prompt.detach().cpu(), args.output_path)
    print(f" Soft prompts saved to {args.output_path}")
