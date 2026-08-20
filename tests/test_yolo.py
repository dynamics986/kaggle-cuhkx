from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from cuhkx_har.data import crop_person, flip_normalized_bbox
from cuhkx_har.yolo import _smooth_boxes, audit_labels, audit_weights, stratified_annotation_sample


def test_bbox_flip_crop_and_fallback_geometry() -> None:
    assert flip_normalized_bbox((0.1, 0.2, 0.4, 0.8)) == (0.6, 0.2, 0.9, 0.8)
    image = Image.new("RGB", (100, 100), "white")
    assert crop_person(image, (0.25, 0.25, 0.75, 0.75), 0.0).size == (50, 50)
    boxes, status = _smooth_boxes([[0.1, 0.2, 0.5, 0.8], None], [0.8, 0.0], 0.25)
    assert boxes[0] is not None and boxes[1] is not None
    assert status == ["detected", "neighbor"]
    assert _smooth_boxes([None], [0.0], 0.25)[1] == ["full_frame"]


def test_annotation_sample_is_deterministic_and_covers_actions(tmp_path) -> None:
    rows = []
    for label in range(40):
        user = "user1" if label % 2 else "user16"
        directory = tmp_path / "Depth_Color" / str(label) / user / "trial"
        directory.mkdir(parents=True)
        Image.new("RGB", (12, 8), "white").save(directory / "frame.jpg")
        rows.append(
            {
                "clip_id": f"{label}/{user}/trial",
                "action_name": str(label),
                "label": label,
                "user": user,
                "Depth_Color": directory.relative_to(tmp_path).as_posix(),
            }
        )
    manifest = pd.DataFrame(rows)
    first = stratified_annotation_sample(manifest, tmp_path, count=40)
    second = stratified_annotation_sample(manifest, tmp_path, count=40)
    assert first["source_path"].tolist() == second["source_path"].tolist()
    assert set(first["label"]) == set(range(40))


def test_label_audit_and_weight_limit(tmp_path) -> None:
    root = tmp_path / "annotations"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    image = root / "images" / "0000.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    (root / "labels" / "0000.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    pd.DataFrame(
        [
            {
                "sample_id": "0000",
                "clip_id": "0/user1/t",
                "user": "user1",
                "image_path": "images/0000.jpg",
                "label_path": "labels/0000.txt",
                "image_sha256": digest,
            }
        ]
    ).to_csv(root / "annotation_manifest.csv", index=False)
    assert audit_labels(root)["status"] == "passed"
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"123")
    assert audit_weights(weight, [weight])["passes"]
    (root / "labels" / "0000.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected YOLO"):
        audit_labels(root)


def test_label_audit_accepts_missing_label_as_negative(tmp_path) -> None:
    root = tmp_path / "annotations"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    image = root / "images" / "0000.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    import hashlib

    pd.DataFrame(
        [
            {
                "sample_id": "0000",
                "clip_id": "0/user1/t",
                "user": "user1",
                "image_path": "images/0000.jpg",
                "label_path": "labels/0000.txt",
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ]
    ).to_csv(root / "annotation_manifest.csv", index=False)
    report = audit_labels(root)
    assert report["positive_frames"] == 0
    assert report["negative_frames"] == 1


def test_label_audit_ignores_labelimg_classes_sidecar(tmp_path) -> None:
    root = tmp_path / "annotations"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    image = root / "images" / "0000.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    (root / "labels" / "0000.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (root / "labels" / "classes.txt").write_text("person\n", encoding="utf-8")
    import hashlib

    pd.DataFrame(
        [
            {
                "sample_id": "0000",
                "clip_id": "0/user1/t",
                "user": "user1",
                "image_path": "images/0000.jpg",
                "label_path": "labels/0000.txt",
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ]
    ).to_csv(root / "annotation_manifest.csv", index=False)
    assert audit_labels(root)["positive_frames"] == 1
