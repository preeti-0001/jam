import argparse
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from conf.conf_parser import parse_conf_file, parse_conf, save_yaml

# ===== IMPORT YOUR MODEL =====
from algorithms.alg import AverageQueryMatching
from data.feature import FeatureHolder
from data.dataset import TrainQueryDataset
from usage.song import get_metadata
import os


class JAMRecommender:
    def __init__(self, model_path, conf, dataset, feature_holder, device="cuda"):
        self.device = device

        # ===== Load Model =====
        self.model = AverageQueryMatching.build_from_conf(conf, dataset, feature_holder)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()

        # ===== Load Language Model (IMPORTANT) =====
        self.tokenizer = AutoTokenizer.from_pretrained(conf["language_model"]["tokenizer_name"])
        self.text_model = AutoModel.from_pretrained(conf["language_model"]["model_name"]).to(
            device
        )
        self.text_model.eval()

        # ===== Metadata =====
        self.n_items = dataset.n_items

        # You MUST replace this with real mapping
        self.track_db = getattr(dataset, "track_metadata", None)

    # ===== TEXT → EMBEDDING =====
    def encode_query(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            q_embed = outputs.last_hidden_state.mean(dim=1)  # (1, lang_dim)

        return q_embed

    # ===== CORE RECOMMENDATION =====
    def recommend(self, text, user_id=0, top_k=5):
        q_text = self.encode_query(text.lower())  # (1, lang_dim)

        # Prepare tensors
        i_idxs = torch.arange(self.n_items).to(self.device)
        u_idxs = torch.tensor([user_id] * self.n_items).to(self.device)
        q_text = q_text.repeat(self.n_items, 1)

        # Inference
        with torch.no_grad():
            scores = self.model(q_text, u_idxs, i_idxs)

        # Top-K
        topk = torch.topk(scores, top_k)

        indices = topk.indices.cpu().tolist()

        # Map to songs (fallback if no metadata)
        if self.track_db:
            return [self.track_db[i] for i in indices]
        else:
            return indices


# ===== SIMPLE TEST =====
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run the JAM Recommender using model')
    parser.add_argument('--path', '-p', type=str, help='Path to the folder containing the trained model .pth file', required=True)
    args = parser.parse_args()
    if not args.path:
        print("Please provide the path to the trained model using --path or -p argument.")
        exit(1)
    if not (os.path.isdir(args.path) and os.path.isfile(os.path.join(args.path, "model.pth"))   and os.path.isfile(os.path.join(args.path, "conf.yml"))):
        print("Invalid model path. Please provide the correct path to the folder containing the trained model.")
        exit(1)

    conf_path = os.path.join(args.path, "conf.yml")
    conf = parse_conf_file(conf_path)
    dataset = TrainQueryDataset(
        data_path="./data/zenodo/processed", 
        lang_model_conf=conf["language_model"],
    )
    feature_holder = FeatureHolder("./data/zenodo/processed")
    recommender = JAMRecommender(
        model_path=os.path.join(args.path, "model.pth"),
        conf=conf,
        dataset=dataset,
        feature_holder=feature_holder
    )
    print("Welcome to the JAM Recommender! Type your music query below. Type -1 as user ID to exit.")
    while True:
        user_id = int(input("Enter your user ID: "))
        if user_id < 0:
            break;
        query = input("Enter your music query: ")
        results = recommender.recommend(query, user_id=user_id)
        print(results)
    print("Bye")

