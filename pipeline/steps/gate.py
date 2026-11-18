"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Literal, TypedDict

from ..common import ERROR, WARN, Issue

Decision = Literal["Release", "Hold", "Rework"]


class Verdict(TypedDict):
    decision: Decision
    n_errors: int
    n_warn_images: int
    n_images: int
    warn_ratio: float
    reasons: list[str]


def count_errors(issues: Iterable[Issue]) -> int:
    """Số lỗi bắt buộc — đếm MỌI lỗi ERROR, kể cả ở ảnh đã duyệt hay cấp dataset."""
    return sum(issue.severity == ERROR for issue in issues)


def warned_images(issues: Iterable[Issue], image_stems: Collection[str], reviewed: Collection[str]) -> set[str]:
    """Các ảnh trong dataset còn ≥ 1 cảnh báo mà người chưa duyệt."""
    return {
        issue.image for issue in issues
        if issue.severity == WARN and issue.image in image_stems and issue.image not in reviewed
    }


def decide(issues: list[Issue], image_stems: Collection[str], cfg: dict[str, Any],
           reviewed: Collection[str] = frozenset()) -> Verdict:
    stems = set(image_stems)
    hold_ratio: float = cfg["gate"]["warn_ratio_hold"]

    n_errors = count_errors(issues)
    n_warn_images = len(warned_images(issues, stems, set(reviewed)))
    n_images = len(stems)
    warn_ratio = round(n_warn_images / n_images, 4) if n_images else 0.0

    ratio_text = f"{n_warn_images}/{n_images} ảnh còn cảnh báo chưa duyệt ({warn_ratio:.1%})"
    if n_errors:
        decision: Decision = "Rework"
        reasons = [f"còn {n_errors} lỗi bắt buộc (ERROR): phải sửa trước khi xuất", ratio_text]
    elif warn_ratio >= hold_ratio:
        decision = "Hold"
        reasons = ["không còn lỗi bắt buộc", f"{ratio_text} >= ngưỡng {hold_ratio:.0%}: chờ người duyệt thêm"]
    else:
        decision = "Release"
        reasons = ["không còn lỗi bắt buộc", f"{ratio_text} < ngưỡng {hold_ratio:.0%}"]

    return Verdict(decision=decision, n_errors=n_errors, n_warn_images=n_warn_images,
                   n_images=n_images, warn_ratio=warn_ratio, reasons=reasons)
