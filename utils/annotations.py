"""Annotation manager - handles category management and YOLO export"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Annotation:
    """Single rectangle annotation stored in pixel coordinates."""
    class_id: int
    x: float  # top-left pixel x
    y: float  # top-left pixel y
    w: float  # width in pixels
    h: float  # height in pixels

    def to_yolo_line(self, img_w: int, img_h: int) -> str:
        """Convert to YOLO format: class_id x_center y_center width height (all normalized)."""
        xc = (self.x + self.w / 2) / img_w
        yc = (self.y + self.h / 2) / img_h
        bw = self.w / img_w
        bh = self.h / img_h
        return f"{self.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


class AnnotationManager:
    """Manages categories and per-image annotations."""

    def __init__(self) -> None:
        self.categories: list[str] = []
        # key=image filename, value=list of Annotation
        self.annotations: dict[str, list[Annotation]] = {}

    # ---- Category management ----

    def add_category(self, name: str) -> int:
        if name in self.categories:
            return self.categories.index(name)
        idx = len(self.categories)
        self.categories.append(name)
        return idx

    def remove_category(self, index: int) -> bool:
        if 0 <= index < len(self.categories):
            self.categories.pop(index)
            # Re-index annotations
            for img_anns in self.annotations.values():
                for a in img_anns:
                    if a.class_id > index:
                        a.class_id -= 1
            return True
        return False

    def rename_category(self, index: int, new_name: str) -> bool:
        if 0 <= index < len(self.categories):
            self.categories[index] = new_name
            return True
        return False

    # ---- Annotation CRUD ----

    def add_annotation(self, filename: str, annotation: Annotation) -> None:
        self.annotations.setdefault(filename, []).append(annotation)

    def remove_annotation(self, filename: str, index: int) -> bool:
        anns = self.annotations.get(filename)
        if anns and 0 <= index < len(anns):
            anns.pop(index)
            return True
        return False

    def get_annotations(self, filename: str) -> list[Annotation]:
        return list(self.annotations.get(filename, []))

    def has_annotations(self, filename: str) -> bool:
        return filename in self.annotations and len(self.annotations[filename]) > 0

    def clear_current(self, filename: str) -> None:
        self.annotations.pop(filename, None)

    # ---- Export ----

    def export_yolo(self, image_folder: str, output_dir: str, val_ratio: float = 0.2) -> str:
        """Export annotations + images to YOLO format.

        Returns path to generated dataset.yaml.
        """
        src = Path(image_folder).resolve()
        dst = Path(output_dir).resolve()

        # Collect all image files
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images = sorted(
            f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )

        if not all_images:
            raise ValueError(f"No supported images found in {image_folder}")

        # Split train/val
        n_val = max(1, int(len(all_images) * val_ratio))
        train_images = all_images[:-n_val] if n_val else all_images
        val_images = all_images[-n_val:] if n_val else []

        # Create output dirs
        for split in ("train", "val"):
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        def export_split(images: list[Path], split: str) -> None:
            for img in images:
                # Copy image
                shutil.copy2(img, dst / "images" / split / img.name)

                # Export annotations
                txt_name = img.with_suffix(".txt").name
                anns = self.annotations.get(img.name, [])
                label_path = dst / "labels" / split / txt_name
                img_w = img.width if hasattr(img, "width") else 0
                # Need actual image dimensions
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(img) as pil:
                        img_w, img_h = pil.size
                except Exception:
                    img_h = 0

                with open(label_path, "w") as f:
                    for ann in anns:
                        if img_w > 0 and img_h > 0:
                            f.write(ann.to_yolo_line(img_w, img_h))

        # We need actual image dims, so do it properly
        from PIL import Image as PILImage
        for split, images in [("train", train_images), ("val", val_images)]:
            for img in images:
                shutil.copy2(img, dst / "images" / split / img.name)
                try:
                    with PILImage.open(img) as pil:
                        img_w, img_h = pil.size
                except Exception:
                    continue

                txt_name = img.with_suffix(".txt").name
                anns = self.annotations.get(img.name, [])
                label_path = dst / "labels" / split / txt_name
                with open(label_path, "w") as f:
                    for ann in anns:
                        f.write(ann.to_yolo_line(img_w, img_h))

        # Generate YAML
        rel_dst = dst.relative_to(dst.parent)
        yaml_lines = [
            f"path: {dst}",
            f"train: {rel_dst / 'images' / 'train'}",
            f"val: {rel_dst / 'images' / 'val'}",
            f"nc: {len(self.categories)}",
            f"names: {self.categories}",
        ]
        yaml_path = dst / "dataset.yaml"
        yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
        return str(yaml_path)
