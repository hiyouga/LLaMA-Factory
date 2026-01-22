import os
import json
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models

class ResNet50Extractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Identity()
        self.model.to(device).eval()
        self.tf = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        feat = self.model(x)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()[0]

def baseline_filter_50(test_samples, image_root, save_path, gallery_size=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ResNet50Extractor(device)

    features = {}
    id_map = {}
    all_img_paths = []
    
    print("[Step 1] Extracting ResNet features and indexing IDs...")
    for s in tqdm(test_samples):
        img_p = s["img_path"]
        pid = os.path.basename(img_p).split("_")[0]
        if pid not in id_map: id_map[pid] = []
        id_map[pid].append(img_p)
        all_img_paths.append(img_p)
        features[img_p] = extractor.extract(os.path.join(image_root, img_p))

    results = {}
    print("[Step 2] Building 50-image fixed gallery per query...")
    for s in tqdm(test_samples):
        q_path = s["img_path"]
        q_pid = os.path.basename(q_path).split("_")[0]
        q_feat = features[q_path]

        # 1. 找到所有同 ID 的正样本（排除自身）
        positives = [p for p in id_map[q_pid] if p != q_path]
        
        # 2. 计算与库中所有图的相似度
        scores = []
        for g_path in all_img_paths:
            if g_path == q_path: continue
            sim = float(np.dot(q_feat, features[g_path]))
            scores.append((g_path, sim))
        
        # 按相似度从高到低排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 组合 Gallery：Positives + 靠前的负样本补充到 50 张
        final_gallery = positives.copy()
        pos_set = set(positives)
        for g_path, _ in scores:
            if len(final_gallery) >= gallery_size: break
            if g_path not in pos_set:
                final_gallery.append(g_path)

        # 将final_gallery随机打乱
        random.shuffle(final_gallery)

        
        results[str(s["id"])] = {"query": q_path, "gallery": final_gallery}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)