"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Với mỗi ảnh có file nhãn và có trong preds:
  P = dự đoán có conf >= conf  VÀ diện tích px >= min_area_px
  Với mỗi dự đoán r trong P, cands = nhãn (MỌI lớp) có IoU(r, nhãn) >= iou_missing:
   (1) cands rỗng                                         -> POSSIBLE_MISSING
   (2) có cands cùng lớp: IoU tốt nhất < iou_loose        -> POSSIBLE_LOOSE_BOX, ngược lại không cờ
   (3) không có cands cùng lớp: IoU tốt nhất >= iou_match -> POSSIBLE_WRONG_CLASS, ngược lại (mơ hồ) không cờ
  Với mỗi dòng nhãn hợp lệ: IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                                          -> POSSIBLE_FALSE_POS (Issue.line = số dòng gốc)
Thứ tự: theo ảnh, theo thứ tự dự đoán, cuối cùng các FALSE_POS theo số dòng. Tất cả là WARN.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..common import WARN, Dataset, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds

CHECK = "second_opinion"

Box = tuple[float, float, float, float]       # (cx, cy, w, h) chuẩn hoá 0–1


@dataclass(frozen=True)
class Prediction:
    cls: int
    conf: float
    box: Box

    @classmethod
    def from_row(cls, row: Sequence[float]) -> Prediction:
        return cls(int(row[0]), float(row[1]), (float(row[2]), float(row[3]), float(row[4]), float(row[5])))

    def area_px(self, image_size: tuple[int, int]) -> float:
        img_w, img_h = image_size
        return self.box[2] * img_w * self.box[3] * img_h


@dataclass(frozen=True)
class Label:
    line: int
    cls: int
    box: Box


@dataclass(frozen=True)
class Thresholds:
    conf: float
    min_area_px: float
    fp_conf: float
    iou_missing: float
    iou_match: float
    iou_loose: float

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> Thresholds:
        so = cfg["second_opinion"]
        return cls(so["conf"], so["min_area_px"], so["fp_conf"], so["iou_missing"], so["iou_match"], so["iou_loose"])


def read_labels(ds: Dataset, stem: str) -> list[Label]:
    """Các dòng nhãn hợp lệ kèm số dòng GỐC; dòng hỏng / id lớp lạ bị bỏ qua (label_format đã báo)."""
    labels: list[Label] = []
    for line_no, text in ds.read_label_lines(stem) or []:
        try:
            cls, cx, cy, w, h = parse_yolo_line(text)
        except ValueError:
            continue
        if 0 <= cls < len(ds.classes):
            labels.append(Label(line_no, cls, (cx, cy, w, h)))
    return labels


def _best_iou(pred: Prediction, labels: list[Label]) -> float:
    return max((iou_xywh(pred.box, label.box) for label in labels), default=0.0)


def judge_prediction(stem: str, pred: Prediction, labels: list[Label], th: Thresholds) -> Issue | None:
    """Một dự đoán đáng tin -> tối đa một issue: MISSING, LOOSE_BOX hoặc WRONG_CLASS."""
    cands = [label for label in labels if iou_xywh(pred.box, label.box) >= th.iou_missing]
    extra: dict[str, Any] = {"conf": pred.conf, "pred_class": pred.cls}
    if not cands:
        return Issue(CHECK, WARN, "POSSIBLE_MISSING", stem,
                     f"model thấy lớp {pred.cls} (conf {pred.conf:.2f}) mà nhãn không có box", extra=extra)

    same_class = [label for label in cands if label.cls == pred.cls]
    if same_class:
        iou = _best_iou(pred, same_class)
        if iou >= th.iou_loose:
            return None
        return Issue(CHECK, WARN, "POSSIBLE_LOOSE_BOX", stem,
                     f"box lớp {pred.cls} lệch/lỏng so với model (IoU {iou:.2f})", extra={**extra, "iou": round(iou, 4)})

    best = max(cands, key=lambda label: iou_xywh(pred.box, label.box))
    iou = iou_xywh(pred.box, best.box)
    if iou < th.iou_match:
        return None                             # mơ hồ: khác lớp, chồng một phần — để FALSE_POS / người duyệt
    return Issue(CHECK, WARN, "POSSIBLE_WRONG_CLASS", stem,
                 f"nhãn lớp {best.cls} nhưng model nói lớp {pred.cls} (IoU {iou:.2f})",
                 extra={**extra, "iou": round(iou, 4), "label_class": best.cls, "label_line": best.line})


def false_positives(stem: str, labels: list[Label], preds: list[Prediction], th: Thresholds) -> list[Issue]:
    """Box nhãn không khớp vật nào model thấy (mọi lớp, conf >= fp_conf, KHÔNG lọc diện tích)."""
    support = [p for p in preds if p.conf >= th.fp_conf]
    issues: list[Issue] = []
    for label in labels:
        iou = max((iou_xywh(label.box, p.box) for p in support), default=0.0)
        if iou < th.iou_missing:
            issues.append(Issue(CHECK, WARN, "POSSIBLE_FALSE_POS", stem,
                                f"box lớp {label.cls} không khớp vật nào model thấy (IoU tốt nhất {iou:.2f})",
                                label.line, {"iou": round(iou, 4)}))
    return issues


def check_image(ds: Dataset, stem: str, rows: list[Sequence[float]], th: Thresholds) -> list[Issue]:
    image_size = ds.image_size(stem)
    if image_size is None:                      # ảnh hỏng: check images đã báo
        return []
    preds = [Prediction.from_row(row) for row in rows]
    labels = read_labels(ds, stem)
    trusted = [p for p in preds if p.conf >= th.conf and p.area_px(image_size) >= th.min_area_px]

    issues = [issue for p in trusted if (issue := judge_prediction(stem, p, labels, th)) is not None]
    issues.extend(false_positives(stem, labels, preds, th))
    return issues


def run(ds: Dataset, cfg: dict[str, Any], preds: dict[str, list[Sequence[float]]] | None = None) -> list[Issue]:
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    th = Thresholds.from_cfg(cfg)
    issues: list[Issue] = []
    for image_file in ds.image_files():
        stem = image_file.stem
        if stem in preds and ds.label_path(stem).exists():
            issues.extend(check_image(ds, stem, preds[stem], th))
    return issues
