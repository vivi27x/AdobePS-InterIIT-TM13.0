from pprint import pprint
import gc
import torch
import torch.ao.quantization
from torch.utils.data import DataLoader
from dataset import CustomDataset
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import parse_arguments, load_config
import json
import os

def main():
    """
    Loads a pretrained model, processes a test dataset, generates predictions for each image, 
    and stores the results in a JSON file.

    Returns:
        None
    """
    gc.collect()
    args = parse_arguments()
    cfg = load_config(args.cfg)
    pprint(cfg)

    # Preliminary setup using the config file
    torch.manual_seed(cfg["inference"]["seed"])
    random.seed(cfg["inference"]["seed"])
    np.random.seed(cfg["inference"]["seed"])
    torch.set_float32_matmul_precision("medium")
    
    ftl_model = torch.jit.load('qftl.pt')

    print(f"Loading Dataset from {cfg['dataset']['path']}")
    test_dataset = CustomDataset(
        dataset_path=cfg["dataset"]["path"],
        resolution=cfg["inference"]["resolution"],
    )

    # Loads the Dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    prediction_list  = []
    with torch.no_grad():
        for batch_num, argz in tqdm(enumerate(test_loader)):
            # Get the output logits
            output = ftl_model(argz['image'])["logits"]
            output = output.squeeze()
            
            # Iterate over each prediction
            for index, val in enumerate(output):
                prediction = "real" if val > 0 else "fake"
                # Extract the corresponding image path
                image_path = argz['image_path'][index]  # Ensure 'path' key exists and matches batch size
                # Append to prediction list
                index = os.path.splitext(os.path.basename(image_path))[0]
                prediction_list.append({
                    "index": int(index),
                    "prediction": prediction, 
                })

    # Convert the prediction list to JSON and write it to a file
    results_json = json.dumps(prediction_list, indent=4)
    print(results_json)
    with open("30_task1.json", "w") as f:
        f.write(results_json)

if __name__ == "__main__":
    main()

