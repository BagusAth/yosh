from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

data_yaml = Path("cassava-v7-standardized/data.yaml").resolve()
train_img_dir = Path("cassava-v7-standardized/train/images").resolve()
valid_img_dir = Path("cassava-v7-standardized/valid/images").resolve()
test_img_dir = Path("cassava-v7-standardized/test/images").resolve()

print("=" * 80)
print("VERIFYING STANDARDIZED DATASET DATALOADER SCANNING")
print("=" * 80)

for split_name, img_dir in [("TRAIN", train_img_dir), ("VALID", valid_img_dir), ("TEST", test_img_dir)]:
    print(f"\n--- Scanning {split_name} Split ---")
    dataset = YOLODataset(
        img_path=str(img_dir),
        data={"names": {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "HEALTHY"}},
        task="detect",
        hyp=DEFAULT_CFG,
    )
    n_images = len(dataset.im_files)
    n_labels = len(dataset.labels)
    print(f"[{split_name}] Total images loaded: {n_images}")
    print(f"[{split_name}] Total label entries : {n_labels}")

    # Check cache details
    cache_path = img_dir.parent / "labels.cache"
    print(f"[{split_name}] Cache path: {cache_path} (Exists: {cache_path.exists()})")
    if cache_path.exists():
        import numpy as np
        cdata = np.load(cache_path, allow_pickle=True).item()
        print(f"[{split_name}] Cache results: {cdata.get('results')}")
        print(f"[{split_name}] Cache msgs count: {len(cdata.get('msgs', []))}")
        if cdata.get('msgs'):
            for m in cdata['msgs'][:5]:
                print(f"    Warning: {m}")

print("\nDATALOADER AUDIT COMPLETE!")
