"""Dataset parser - handles YOLO txt and COCO json formats"""

import json
from pathlib import Path
from collections import Counter


class DatasetParser:
    def __init__(self):
        self.supported_formats = {
            "yolo": [".txt"],
            "coco": [".json"],
        }

    def parse_and_validate(self, dataset_dir: str) -> dict:
        """Parse dataset directory and return statistics"""
        dataset_path = Path(dataset_dir)
        result = {}

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        train_images = images_dir / "train" if (images_dir / "train").exists() else images_dir
        val_images = images_dir / "val" if (images_dir / "val").exists() else None
        train_labels = labels_dir / "train" if (labels_dir / "train").exists() else labels_dir
        val_labels = labels_dir / "val" if (labels_dir / "val").exists() else None

        # Count images
        train_imgs = list(train_images.glob("*")) if train_images.exists() else []
        train_imgs = [f for f in train_imgs if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        result["train_images"] = len(train_imgs)

        if val_images:
            val_imgs = [f for f in val_images.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            result["val_images"] = len(val_imgs)
        else:
            result["val_images"] = 0

        result["total_images"] = len(train_imgs) + result["val_images"]

        # Detect classes from label files
        labels_dir_to_check = train_labels if train_labels.exists() else labels_dir
        if labels_dir_to_check.exists():
            label_files = list(labels_dir_to_check.glob("*.txt"))
            result["label_files"] = len(label_files)

            all_classes = []
            for lf in label_files[:100]:
                try:
                    with open(lf, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                all_classes.append(cls_id)
                except (ValueError, IndexError):
                    continue

            if all_classes:
                counter = Counter(all_classes)
                result["classes"] = sorted(counter.keys())
                result["class_distribution"] = dict(counter)
            else:
                result["classes"] = []
                result["class_distribution"] = {}
        else:
            result["label_files"] = 0
            result["classes"] = []
            result["class_distribution"] = {}

        # Check for YAML config
        coco_yaml = dataset_path / "dataset.yaml"
        if not coco_yaml.exists():
            coco_yaml = dataset_path / "coco.yaml"
        if coco_yaml.exists():
            result["config_file"] = str(coco_yaml)
            result["format"] = "yaml_config"

        # Validation
        if result["total_images"] == 0:
            result["valid"] = False
            result["error"] = "No images found. Expected folder structure:"
            result["hint"] = "images/train/, images/val/, labels/train/, labels/val/"
        elif len(result["classes"]) == 0:
            result["valid"] = False
            result["error"] = "No classes detected. Check label files."
        else:
            result["valid"] = True

        return result

    def create_yaml_config(self, dataset_dir: str, num_classes: int, class_names: list[str]) -> str:
        """Create a dataset.yaml config file for YOLO training"""
        dataset_path = Path(dataset_dir).resolve()
        images_path = dataset_path / "images"
        labels_path = dataset_path / "labels"

        train_img = images_path / "train"
        val_img = images_path / "val" if (images_path / "val").exists() else train_img

        # Use relative paths for compatibility
        rel_train = train_img.relative_to(dataset_path)
        rel_val = val_img.relative_to(dataset_path)

        yaml_content = f"path: {dataset_path}\n"
        yaml_content += f"train: {rel_train}\n"
        yaml_content += f"val: {rel_val}\n"
        yaml_content += f"nc: {num_classes}\n"
        yaml_content += f"names: {class_names}\n"

        yaml_path = dataset_path / "dataset.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        return str(yaml_path)
