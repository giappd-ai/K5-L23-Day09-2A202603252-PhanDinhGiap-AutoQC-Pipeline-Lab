"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) dừng ở đó, không kiểm tiếp hình học.
Mọi ngưỡng đọc từ `cfg["label_format"]`.
"""
from __future__ import annotations

from typing import Any

from ..common import ERROR, WARN, Dataset, Issue, iou_xywh, parse_yolo_line

CHECK = "label_format"

YoloBox = tuple[int, float, float, float, float]      # (id lớp, cx, cy, w, h)


def overflow_amount(cx: float, cy: float, w: float, h: float) -> float:
    """Mức tràn lớn nhất của 4 mép box ra ngoài [0, 1]; 0 nếu box nằm trong ảnh."""
    left, right = cx - w / 2, cx + w / 2
    top, bottom = cy - h / 2, cy + h / 2
    return max(0.0, -left, right - 1.0, -top, bottom - 1.0)


def _parse_or_issue(stem: str, line_no: int, text: str, n_classes: int) -> YoloBox | Issue:
    """Trả box hợp lệ về cú pháp, hoặc Issue ERROR nếu dòng hỏng (BAD_FORMAT / BAD_CLASS_ID / BOX_ZERO)."""
    try:
        box = parse_yolo_line(text)
    except ValueError as exc:
        return Issue(CHECK, ERROR, "BAD_FORMAT", stem, f"dòng sai định dạng: {exc}", line_no)
    cls, _, _, w, h = box
    if not 0 <= cls < n_classes:
        return Issue(CHECK, ERROR, "BAD_CLASS_ID", stem, f"id lớp {cls} ngoài 0..{n_classes - 1}", line_no,
                     {"class_id": cls})
    if w <= 0 or h <= 0:
        return Issue(CHECK, ERROR, "BOX_ZERO", stem, f"box rộng/cao bằng 0 (w={w}, h={h})", line_no)
    return box


def _geometry_issues(stem: str, line_no: int, box: YoloBox, earlier: list[YoloBox],
                     image_size: tuple[int, int] | None, rules: dict[str, Any]) -> list[Issue]:
    """Các lỗi hình học của một box đã đúng cú pháp, so với các box hợp lệ ở dòng trước."""
    cls, cx, cy, w, h = box
    issues: list[Issue] = []

    overflow = overflow_amount(cx, cy, w, h)
    if overflow > rules["overflow_tol"]:
        issues.append(Issue(CHECK, ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn ra ngoài ảnh {overflow:.4f}", line_no,
                            {"overflow": round(overflow, 6)}))

    duplicate_iou = max((iou_xywh(box[1:], prev[1:]) for prev in earlier if prev[0] == cls), default=0.0)
    if duplicate_iou > rules["dup_box_iou"]:
        issues.append(Issue(CHECK, ERROR, "DUP_BOX", stem, f"box lặp với dòng trước (IoU {duplicate_iou:.3f})", line_no,
                            {"iou": round(duplicate_iou, 4)}))

    if image_size is not None:
        img_w, img_h = image_size
        short_side_px = min(w * img_w, h * img_h)
        if short_side_px < rules["min_box_px"]:
            issues.append(Issue(CHECK, WARN, "BOX_TINY", stem, f"cạnh ngắn {short_side_px:.1f}px < {rules['min_box_px']}px",
                                line_no, {"short_side_px": round(short_side_px, 2)}))

    area_frac = w * h
    if area_frac > rules["max_box_frac"]:
        issues.append(Issue(CHECK, WARN, "BOX_HUGE", stem, f"box chiếm {area_frac:.0%} ảnh", line_no,
                            {"area_frac": round(area_frac, 4)}))
    return issues


def check_file(ds: Dataset, stem: str, rules: dict[str, Any]) -> list[Issue]:
    """Kiểm mọi dòng của một file nhãn; số dòng giữ nguyên số dòng gốc (kể cả dòng trống)."""
    issues: list[Issue] = []
    valid_boxes: list[YoloBox] = []
    n_classes = len(ds.classes)
    for line_no, text in ds.read_label_lines(stem) or []:
        parsed = _parse_or_issue(stem, line_no, text, n_classes)
        if isinstance(parsed, Issue):
            issues.append(parsed)
            continue
        issues.extend(_geometry_issues(stem, line_no, parsed, valid_boxes, ds.image_size(stem), rules))
        valid_boxes.append(parsed)
    return issues


def run(ds: Dataset, cfg: dict[str, Any]) -> list[Issue]:
    rules = cfg["label_format"]
    issues: list[Issue] = []
    for label_file in ds.label_files():
        issues.extend(check_file(ds, label_file.stem, rules))
    return issues
