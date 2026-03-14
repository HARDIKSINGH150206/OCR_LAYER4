from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2


@dataclass(frozen=True)
class SplitStats:
    split: str
    total_rows: int
    prepared_images: int
    missing_images: int
    invalid_boxes: int


@dataclass(frozen=True)
class PreparedSample:
    source_image: Path
    output_name: str
    boxes: Tuple[Tuple[float, float, float, float], ...]
    require_positive_box: bool = False


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _candidate_file(primary: Path, template: Path) -> Path:
    if primary.exists():
        return primary
    if template.exists():
        return template
    raise FileNotFoundError(f"Missing file: {primary} (template not found at {template})")


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_name_list(path: Path) -> List[str]:
    names: List[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        name = raw_line.strip()
        if not name:
            continue
        names.append(name)
    return names


def _parse_box_list(raw_value: str) -> List[Tuple[float, float, float, float]]:
    value = (raw_value or "").strip()
    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, (list, tuple)) and len(parsed) == 4 and all(
        isinstance(x, (int, float)) for x in parsed
    ):
        return [(float(parsed[0]), float(parsed[1]), float(parsed[2]), float(parsed[3]))]

    if isinstance(parsed, (list, tuple)) and parsed and all(isinstance(x, (list, tuple)) for x in parsed):
        boxes: List[Tuple[float, float, float, float]] = []
        for box in parsed:
            if len(box) != 4 or not all(isinstance(x, (int, float)) for x in box):
                continue
            boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
        return boxes

    return []


def _clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _to_yolo_line(box: Tuple[float, float, float, float], width: int, height: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return f"{class_id} {cx / width:.6f} {cy / height:.6f} {w / width:.6f} {h / height:.6f}"


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _load_image_as_bgr(src: Path):
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    channels = image.shape[2]
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 3:
        return image
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels > 4:
        return image[:, :, :3]
    return None


def _write_prepared_image(dst: Path, image) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), image))


def _build_label_lookup(label_rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_row_id: Dict[str, Dict[str, str]] = {}
    by_image_name: Dict[str, Dict[str, str]] = {}

    for index, row in enumerate(label_rows):
        row_with_meta = dict(row)
        row_with_meta["__row_index__"] = str(index)

        key_candidates = {
            str(index),
            str(row.get("label_row_id", "")).strip(),
            str(row.get("id", "")).strip(),
            str(row.get("row_id", "")).strip(),
        }
        for key in key_candidates:
            if key:
                by_row_id[key] = row_with_meta

        image_name = str(row.get("image_name", "")).strip()
        if image_name:
            by_image_name[image_name] = row_with_meta

    return by_row_id, by_image_name


def _prepare_split(
    split: str,
    manifest_rows: Sequence[Dict[str, str]],
    images_root: Path,
    labels_by_row_id: Dict[str, Dict[str, str]],
    labels_by_image_name: Dict[str, Dict[str, str]],
    yolo_root: Path,
) -> SplitStats:
    images_out = yolo_root / "images" / split
    labels_out = yolo_root / "labels" / split

    prepared_images = 0
    missing_images = 0
    invalid_boxes = 0

    for row in manifest_rows:
        image_name = str(row.get("image_name", "")).strip()
        if not image_name:
            continue

        src_image = images_root / image_name
        if not src_image.exists():
            missing_images += 1
            continue

        image = _load_image_as_bgr(src_image)
        if image is None:
            missing_images += 1
            continue

        dst_image = images_out / image_name
        if not _write_prepared_image(dst_image, image):
            missing_images += 1
            continue
        height, width = image.shape[:2]

        label_row = None
        label_row_id = str(row.get("label_row_id", "")).strip()
        if label_row_id:
            label_row = labels_by_row_id.get(label_row_id)
        if label_row is None:
            label_row = labels_by_image_name.get(image_name)

        boxes: List[Tuple[float, float, float, float]] = []
        if label_row is not None:
            boxes = _parse_box_list(str(label_row.get("manipulated_region_bbox", "")))

        yolo_lines: List[str] = []
        for box in boxes:
            clipped = _clip_box(box, width=width, height=height)
            if clipped is None:
                invalid_boxes += 1
                continue
            yolo_lines.append(_to_yolo_line(clipped, width=width, height=height, class_id=0))

        label_file = labels_out / (Path(image_name).stem + ".txt")
        _write_lines(label_file, yolo_lines)

        prepared_images += 1

    return SplitStats(
        split=split,
        total_rows=len(manifest_rows),
        prepared_images=prepared_images,
        missing_images=missing_images,
        invalid_boxes=invalid_boxes,
    )


def _mask_to_bbox(mask_path: Path) -> Tuple[float, float, float, float] | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    non_zero = cv2.findNonZero(mask)
    if non_zero is None:
        return None

    x, y, width, height = cv2.boundingRect(non_zero)
    if width <= 0 or height <= 0:
        return None
    return float(x), float(y), float(x + width), float(y + height)


def _hash_split_bucket(name: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 4294967295.0
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _ensure_minimum_split_coverage(
    split_map: Dict[str, List[PreparedSample]],
) -> Dict[str, List[PreparedSample]]:
    if not split_map["train"] and split_map["test"]:
        split_map["train"].append(split_map["test"].pop())

    if not split_map["val"]:
        donor: str | None = None
        if len(split_map["train"]) > 1:
            donor = "train"
        elif split_map["test"]:
            donor = "test"
        if donor is not None:
            split_map["val"].append(split_map[donor].pop())

    return split_map


def _prepare_split_from_samples(split: str, samples: Sequence[PreparedSample], yolo_root: Path) -> SplitStats:
    images_out = yolo_root / "images" / split
    labels_out = yolo_root / "labels" / split

    prepared_images = 0
    missing_images = 0
    invalid_boxes = 0

    for sample in samples:
        src_image = sample.source_image
        if not src_image.exists():
            missing_images += 1
            continue

        image = _load_image_as_bgr(src_image)
        if image is None:
            missing_images += 1
            continue

        dst_image = images_out / sample.output_name
        if not _write_prepared_image(dst_image, image):
            missing_images += 1
            continue

        height, width = image.shape[:2]
        boxes = list(sample.boxes)
        if sample.require_positive_box and not boxes:
            boxes = [(0.0, 0.0, float(width), float(height))]

        yolo_lines: List[str] = []
        for box in boxes:
            clipped = _clip_box(box, width=width, height=height)
            if clipped is None:
                invalid_boxes += 1
                continue
            yolo_lines.append(_to_yolo_line(clipped, width=width, height=height, class_id=0))

        label_file = labels_out / Path(sample.output_name).with_suffix(".txt")
        _write_lines(label_file, yolo_lines)

        prepared_images += 1

    return SplitStats(
        split=split,
        total_rows=len(samples),
        prepared_images=prepared_images,
        missing_images=missing_images,
        invalid_boxes=invalid_boxes,
    )


def _build_fallback_splits(samples: Sequence[PreparedSample]) -> Dict[str, List[PreparedSample]]:
    split_map: Dict[str, List[PreparedSample]] = {"train": [], "val": [], "test": []}

    positive_samples = [sample for sample in samples if sample.require_positive_box]
    negative_samples = [sample for sample in samples if not sample.require_positive_box]

    for group in (positive_samples, negative_samples):
        for sample in group:
            split = _hash_split_bucket(sample.output_name)
            split_map[split].append(sample)

    return _ensure_minimum_split_coverage(split_map)


def _discover_casia_samples(dataset_root: Path) -> List[PreparedSample]:
    casia_root = dataset_root / "CASIA2.0_revised"
    tampered_dir = casia_root / "Tp"
    authentic_dir = casia_root / "Au"
    groundtruth_dir = dataset_root / "CASIA2.0_Groundtruth"

    if not tampered_dir.exists() or not authentic_dir.exists() or not groundtruth_dir.exists():
        return []

    tampered_lookup = {
        entry.name.lower(): entry
        for entry in tampered_dir.iterdir()
        if entry.is_file() and _is_image_file(entry)
    }
    authentic_lookup = {
        entry.name.lower(): entry
        for entry in authentic_dir.iterdir()
        if entry.is_file() and _is_image_file(entry)
    }
    mask_lookup = {
        entry.stem.lower(): entry
        for entry in groundtruth_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".png"
    }

    tampered_list_path = casia_root / "tp_list.txt"
    if tampered_list_path.exists():
        tampered_names = _read_name_list(tampered_list_path)
    else:
        tampered_names = sorted(tampered_lookup.keys())

    authentic_list_path = casia_root / "au_list.txt"
    if authentic_list_path.exists():
        authentic_names = _read_name_list(authentic_list_path)
    else:
        authentic_names = sorted(authentic_lookup.keys())

    samples: List[PreparedSample] = []
    seen_output_names: set[str] = set()

    for image_name in tampered_names:
        source_image = tampered_dir / image_name
        if not source_image.exists():
            fallback_image = tampered_lookup.get(image_name.lower())
            if fallback_image is None:
                continue
            source_image = fallback_image

        if not _is_image_file(source_image):
            continue

        mask_key = f"{source_image.stem}_gt".lower()
        mask_path = mask_lookup.get(mask_key)
        boxes: Tuple[Tuple[float, float, float, float], ...] = ()
        if mask_path is not None:
            bbox = _mask_to_bbox(mask_path)
            if bbox is not None:
                boxes = (bbox,)

        output_name = f"casia_tp_{source_image.name}"
        if output_name in seen_output_names:
            continue
        seen_output_names.add(output_name)
        samples.append(
            PreparedSample(
                source_image=source_image,
                output_name=output_name,
                boxes=boxes,
                require_positive_box=True,
            )
        )

    for image_name in authentic_names:
        source_image = authentic_dir / image_name
        if not source_image.exists():
            fallback_image = authentic_lookup.get(image_name.lower())
            if fallback_image is None:
                continue
            source_image = fallback_image

        if not _is_image_file(source_image):
            continue

        output_name = f"casia_au_{source_image.name}"
        if output_name in seen_output_names:
            continue
        seen_output_names.add(output_name)
        samples.append(
            PreparedSample(
                source_image=source_image,
                output_name=output_name,
                boxes=(),
                require_positive_box=False,
            )
        )

    return samples


def _discover_micc_samples(dataset_root: Path) -> List[PreparedSample]:
    micc_root = dataset_root / "MICC-F220"
    groundtruth_file = micc_root / "groundtruthDB_220.txt"
    if not micc_root.exists() or not groundtruth_file.exists():
        return []

    samples: List[PreparedSample] = []
    seen_output_names: set[str] = set()

    for raw_line in groundtruth_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        image_name = parts[0]
        label_token = parts[-1]
        if label_token not in {"0", "1"}:
            continue

        source_image = micc_root / image_name
        if not source_image.exists() or not _is_image_file(source_image):
            continue

        output_name = f"micc_{source_image.name}"
        if output_name in seen_output_names:
            continue
        seen_output_names.add(output_name)

        samples.append(
            PreparedSample(
                source_image=source_image,
                output_name=output_name,
                boxes=(),
                require_positive_box=(label_token == "1"),
            )
        )

    return samples


def _prepare_manifest_dataset(dataset_root: Path, labels_csv_arg: Path | None, yolo_root: Path) -> List[SplitStats]:
    images_root = dataset_root / "raw" / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Image folder not found: {images_root}")

    labels_csv = _resolve_labels_csv(dataset_root, explicit_path=labels_csv_arg)
    label_rows = _read_csv_rows(labels_csv)
    labels_by_row_id, labels_by_image_name = _build_label_lookup(label_rows)

    if not labels_by_row_id and not labels_by_image_name:
        raise RuntimeError(f"No label rows found in: {labels_csv}")

    stats: List[SplitStats] = []
    for split in ("train", "val", "test"):
        manifest_path = _resolve_manifest(dataset_root, split)
        manifest_rows = _read_csv_rows(manifest_path)
        stats.append(
            _prepare_split(
                split=split,
                manifest_rows=manifest_rows,
                images_root=images_root,
                labels_by_row_id=labels_by_row_id,
                labels_by_image_name=labels_by_image_name,
                yolo_root=yolo_root,
            )
        )

    return stats


def _prepare_fallback_dataset(dataset_root: Path, yolo_root: Path) -> Tuple[List[SplitStats], str]:
    source_details: List[str] = []
    all_samples: List[PreparedSample] = []

    casia_samples = _discover_casia_samples(dataset_root)
    if casia_samples:
        all_samples.extend(casia_samples)
        source_details.append(f"CASIA2.0 ({len(casia_samples)} samples)")

    micc_samples = _discover_micc_samples(dataset_root)
    if micc_samples:
        all_samples.extend(micc_samples)
        source_details.append(f"MICC-F220 ({len(micc_samples)} samples)")

    if not all_samples:
        raise FileNotFoundError(
            "No supported fallback datasets found. Expected CASIA2.0_revised + CASIA2.0_Groundtruth "
            "and/or MICC-F220 under dataset root."
        )

    split_map = _build_fallback_splits(all_samples)
    stats = [
        _prepare_split_from_samples(split=split, samples=split_map[split], yolo_root=yolo_root)
        for split in ("train", "val", "test")
    ]
    return stats, ", ".join(source_details)


def _write_dataset_yaml(yolo_root: Path) -> Path:
    yaml_path = yolo_root / "layer4_yolo_dataset.yaml"
    content = (
        f"path: {yolo_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: expiry_region\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _resolve_manifest(dataset_root: Path, split: str) -> Path:
    split_dir = dataset_root / "splits" / split
    primary = split_dir / f"{split}_manifest.csv"
    template = split_dir / f"{split}_manifest_template.csv"
    return _candidate_file(primary, template)


def _resolve_labels_csv(dataset_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"labels CSV not found: {explicit_path}")
        return explicit_path

    labels_dir = dataset_root / "raw" / "labels"
    primary = labels_dir / "labels.csv"
    template = labels_dir / "labels_template.csv"
    return _candidate_file(primary, template)


def _print_stats(stats: Iterable[SplitStats], yaml_path: Path) -> None:
    print("Prepared YOLO dataset")
    print(f"YAML: {yaml_path}")
    for split_stat in stats:
        print(
            f"- {split_stat.split}: rows={split_stat.total_rows}, "
            f"prepared={split_stat.prepared_images}, missing_images={split_stat.missing_images}, "
            f"invalid_boxes={split_stat.invalid_boxes}"
        )


def _resolve_workers(raw_workers: int) -> int:
    if raw_workers > 0:
        return raw_workers
    cpu_total = os.cpu_count() or 8
    return max(2, min(16, cpu_total - 2))


def _resolve_batch(raw_batch: str) -> int:
    value = raw_batch.strip().lower()
    if value in {"auto", "-1", "max", "0"}:
        return -1

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("Invalid batch value. Use an integer or 'auto'.") from exc

    if parsed == 0:
        return -1
    if parsed < -1:
        raise ValueError("Invalid batch value. Use a positive integer, 0, -1, or 'auto'.")
    return parsed


def _resolve_imgsz(
    raw_imgsz: str,
    torch_module,
    selected_indices: Sequence[int],
) -> Tuple[int, float | None]:
    value = raw_imgsz.strip().lower()
    if value not in {"auto", "-1", "0", "max"}:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("Invalid imgsz value. Use a positive integer or 'auto'.") from exc
        if parsed <= 0:
            raise ValueError("Invalid imgsz value. Use a positive integer or 'auto'.")
        aligned = max(320, (parsed // 32) * 32)
        return aligned, None

    free_gib_values: List[float] = []
    for index in selected_indices:
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(index)
            free_gib_values.append(float(free_bytes) / (1024**3))
        except Exception:
            total_bytes = float(torch_module.cuda.get_device_properties(index).total_memory)
            free_gib_values.append((total_bytes * 0.7) / (1024**3))

    min_free_gib = min(free_gib_values) if free_gib_values else 0.0
    if min_free_gib >= 22:
        resolved = 1280
    elif min_free_gib >= 16:
        resolved = 1024
    elif min_free_gib >= 12:
        resolved = 960
    elif min_free_gib >= 8:
        resolved = 768
    elif min_free_gib >= 6:
        resolved = 640
    else:
        resolved = 512

    resolved = max(320, (resolved // 32) * 32)
    return resolved, min_free_gib


def _resolve_cache_mode(raw_cache: str) -> str | bool:
    value = raw_cache.strip().lower()
    if value == "ram":
        return "ram"
    if value == "disk":
        return "disk"
    if value in {"none", "false", "off", "0"}:
        return False
    raise ValueError("Invalid cache mode. Use one of: ram, disk, none")


def _resolve_cuda_device(device_arg: str, cuda_device_count: int) -> str:
    if cuda_device_count <= 0:
        raise RuntimeError("No CUDA devices detected")

    device = device_arg.strip().lower().replace(" ", "")
    if not device:
        device = "0"

    if device in {"cpu", "mps", "xpu"}:
        raise RuntimeError("CPU/MPS/XPU device selection is disabled for training. Use CUDA device index, e.g. 0")

    if device == "cuda":
        device = "0"
    elif device.startswith("cuda:"):
        device = device.split("cuda:", 1)[1]

    if device == "all":
        return ",".join(str(index) for index in range(cuda_device_count))

    parts = device.split(",")
    if not all(part.isdigit() for part in parts):
        raise RuntimeError(
            "Invalid --device value. Use CUDA index format like '0' or '0,1' (or 'cuda:0', 'all')."
        )

    for part in parts:
        index = int(part)
        if index < 0 or index >= cuda_device_count:
            raise RuntimeError(
                f"Requested CUDA device {index} is out of range (available: 0..{cuda_device_count - 1})."
            )

    return ",".join(parts)


def _configure_cuda_for_speed(torch_module) -> None:
    torch_module.backends.cudnn.benchmark = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cuda.matmul.allow_tf32 = True
    try:
        torch_module.set_float32_matmul_precision("high")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Layer-4 OCR dataset")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=Path("dataset") / "prepared_yolo")
    parser.add_argument("--output-root", type=Path, default=Path("models") / "yolo_finetune")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=str, default="auto", help="Input size integer or 'auto' for GPU memory-aware sizing")
    parser.add_argument("--batch", type=str, default="auto", help="Batch size integer or 'auto' for CUDA auto-batch")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers (0 = auto-tune)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index/list only, e.g. 0 or 0,1")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision")
    parser.add_argument("--cache", type=str, default="ram", choices=("ram", "disk", "none"), help="Dataset cache mode")
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True, help="Use cosine LR schedule")
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True, help="Run validation during training")
    parser.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=True, help="Use throughput-oriented train settings")
    parser.add_argument("--close-mosaic", type=int, default=5, help="Disable mosaic augmentation after N final epochs")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save interval in epochs (-1 disables periodic saves)")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="Save training plots")
    parser.add_argument("--name", type=str, default="layer4_expiry_region")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()

    yolo_root = args.prepared_root.resolve()
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    (yolo_root / "images").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels").mkdir(parents=True, exist_ok=True)

    stats: List[SplitStats] | None = None
    dataset_source = ""
    manifest_error: Exception | None = None

    if (dataset_root / "raw" / "images").exists():
        try:
            stats = _prepare_manifest_dataset(dataset_root=dataset_root, labels_csv_arg=args.labels_csv, yolo_root=yolo_root)
            dataset_source = "manifest files under dataset/raw + dataset/splits"
        except (FileNotFoundError, RuntimeError) as exc:
            manifest_error = exc
            print(f"Manifest dataset preparation failed: {exc}")
            print("Trying fallback discovery from CASIA/MICC folders...")

    if stats is None:
        try:
            stats, fallback_source = _prepare_fallback_dataset(dataset_root=dataset_root, yolo_root=yolo_root)
            dataset_source = f"auto-discovered from {fallback_source}"
        except FileNotFoundError as fallback_error:
            if manifest_error is not None:
                raise manifest_error
            raise fallback_error

    yaml_path = _write_dataset_yaml(yolo_root)
    print(f"Dataset source: {dataset_source}")
    _print_stats(stats, yaml_path)

    train_prepared = next(item for item in stats if item.split == "train").prepared_images
    val_prepared = next(item for item in stats if item.split == "val").prepared_images
    if args.dry_run:
        if train_prepared == 0 or val_prepared == 0:
            print(
                "Dry run completed: dataset is not ready for training yet. "
                "Provide manifest files under dataset/raw + dataset/splits, or place "
                "CASIA2.0_revised + CASIA2.0_Groundtruth and/or MICC-F220 under dataset root."
            )
        else:
            print("Dry run completed. Dataset conversion is ready.")
        return

    if train_prepared == 0 or val_prepared == 0:
        raise RuntimeError(
            "Insufficient data for fine-tuning. Provide manifest files under dataset/raw + "
            "dataset/splits, or place CASIA2.0_revised + CASIA2.0_Groundtruth and/or MICC-F220 "
            "under dataset root."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but torch.cuda.is_available() is False.")

    cuda_device_count = torch.cuda.device_count()
    resolved_device = _resolve_cuda_device(args.device, cuda_device_count)
    resolved_workers = _resolve_workers(args.workers)
    resolved_batch = _resolve_batch(args.batch)
    resolved_cache = _resolve_cache_mode(args.cache)

    _configure_cuda_for_speed(torch)

    selected_indices = [int(value) for value in resolved_device.split(",")]
    resolved_imgsz, min_free_gib = _resolve_imgsz(args.imgsz, torch, selected_indices)
    selected_names = [torch.cuda.get_device_name(index) for index in selected_indices]
    print(
        "CUDA acceleration enabled "
        f"| device={resolved_device} | workers={resolved_workers} | batch={resolved_batch} "
        f"| imgsz={resolved_imgsz} "
        f"| amp={args.amp} | cache={resolved_cache} | fast_mode={args.fast_mode}"
    )
    if min_free_gib is not None:
        print(f"  - auto-imgsz selected from minimum free GPU memory: {min_free_gib:.2f} GiB")
    for index, name in zip(selected_indices, selected_names):
        print(f"  - cuda:{index}: {name}")

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": resolved_imgsz,
        "batch": resolved_batch,
        "workers": resolved_workers,
        "device": resolved_device,
        "amp": args.amp,
        "cache": resolved_cache,
        "cos_lr": args.cos_lr,
        "val": args.val,
        "close_mosaic": max(0, args.close_mosaic),
        "save_period": args.save_period,
        "deterministic": False,
        "plots": args.plots,
        "project": str(args.output_root.resolve()),
        "name": args.name,
        "exist_ok": True,
    }

    if args.fast_mode:
        train_kwargs.update(
            {
                "mosaic": 0.5,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "degrees": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "erasing": 0.0,
            }
        )

    model.train(**train_kwargs)

    best_weights = args.output_root.resolve() / args.name / "weights" / "best.pt"
    print(f"Fine-tuning completed. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
