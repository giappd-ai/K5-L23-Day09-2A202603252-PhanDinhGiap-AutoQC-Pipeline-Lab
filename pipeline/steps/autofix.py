"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Cách làm: xử lý lỗi cấp ảnh/dataset trước (xoá ảnh, đổi tên lớp, chuyển người gán), rồi gom lỗi dòng theo file,
đọc cả file (kể cả dòng trống, để số dòng gốc không lệch), quyết định từng dòng và ghi lại MỘT lần.
Một dòng vừa phải xoá vừa phải cắt -> xoá thắng. Chỉ sửa `ds`, không động tới WARN; chạy lại là idempotent.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

import yaml

from ..common import ERROR, Dataset, Issue, parse_yolo_line

DROP_LINE_CODES = frozenset({"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"})
CLIP_LINE_CODES = frozenset({"BOX_OUT_OF_IMAGE"})
LINE_CODES = DROP_LINE_CODES | CLIP_LINE_CODES
REMOVE_IMAGE_CODES = frozenset({"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"})


class FixRecord(TypedDict):
    image: str
    line: int | None
    action: str
    before: str
    after: str


def _record(image: str, action: str, before: str, after: str, line: int | None = None) -> FixRecord:
    return FixRecord(image=image, line=line, action=action, before=before, after=after)


# ---------------------------------------------------------------- lỗi cấp ảnh / dataset
def remove_image(ds: Dataset, stem: str) -> FixRecord | None:
    """Xoá ảnh và file nhãn của nó; None nếu ảnh đã không còn (idempotent)."""
    image_path = ds.image_path(stem)
    if image_path is None:
        return None
    image_path.unlink()
    label_path = ds.label_path(stem)
    if label_path.exists():
        label_path.unlink()
    return _record(stem, "remove_image", image_path.name, "(đã xoá ảnh + nhãn)")


def rename_classes(ds: Dataset, classes: list[str]) -> FixRecord | None:
    """Ghi lại `names` trong data.yaml theo đúng danh sách lớp dự án; None nếu đã đúng."""
    old_names = ds.yaml_names()
    new_names = dict(enumerate(classes))
    if old_names == new_names:
        return None
    content: dict[str, Any] = {}
    if ds.yaml_path.exists():
        content = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
    content["names"] = new_names
    ds.yaml_path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _record("data.yaml", "rename_classes", str(old_names), str(new_names))


def needs_label(stem: str) -> FixRecord:
    """Ảnh chưa có nhãn: KHÔNG tạo file (rỗng nghĩa là "ảnh nền"), chỉ ghi lại để chuyển người gán."""
    return _record(stem, "needs_label", "(không có file)", "chuyển người gán")


# ---------------------------------------------------------------- lỗi dòng nhãn
def clip_line(text: str) -> str | None:
    """Cắt box về trong [0, 1], giữ lớp. None nếu sau khi cắt box không còn diện tích (hoặc dòng không parse được)."""
    try:
        cls, cx, cy, w, h = parse_yolo_line(text)
    except ValueError:
        return None
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def fix_label_file(ds: Dataset, stem: str, codes_by_line: dict[int, set[str]]) -> list[FixRecord]:
    """Sửa mọi dòng lỗi của một file theo số dòng GỐC rồi ghi lại một lần."""
    label_path = ds.label_path(stem)
    if not label_path.exists():                 # ảnh/nhãn đã bị xoá ở bước trước: bỏ qua êm
        return []
    lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept: list[str] = []
    fixes: list[FixRecord] = []
    for line_no, text in enumerate(lines, start=1):
        codes = codes_by_line.get(line_no, set())
        if codes & DROP_LINE_CODES:
            fixes.append(_record(stem, "drop_line", text, "(đã xoá)", line_no))
            continue
        if codes & CLIP_LINE_CODES:
            clipped = clip_line(text)
            if clipped is None:
                fixes.append(_record(stem, "drop_line", text, "(đã xoá: box nằm ngoài ảnh)", line_no))
                continue
            fixes.append(_record(stem, "clip_box", text, clipped, line_no))
            text = clipped
        kept.append(text)
    if fixes:
        label_path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return fixes


# ---------------------------------------------------------------- điểm vào
def _fix_image_level(ds: Dataset, issue: Issue, cfg: dict[str, Any]) -> FixRecord | None:
    """Sửa một lỗi cấp ảnh/dataset; None nếu không có gì để làm."""
    if issue.code in REMOVE_IMAGE_CODES:
        return remove_image(ds, issue.image)
    if issue.code == "BAD_CLASS_NAME":
        return rename_classes(ds, cfg["classes"])
    if issue.code == "MISSING_LABEL_FILE":
        return needs_label(issue.image)
    return None


def fix(ds: Dataset, issues: list[Issue], cfg: dict[str, Any]) -> list[FixRecord]:
    fixes: list[FixRecord] = []
    done: set[tuple[str, str]] = set()          # (ảnh, nhóm sửa): nhiều issue cùng loại trên một ảnh chỉ sửa một lần
    line_codes: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for issue in (i for i in issues if i.severity == ERROR):
        if issue.code in LINE_CODES:
            if issue.line is not None:
                line_codes[issue.image][issue.line].add(issue.code)
            continue
        key = (issue.image, "remove" if issue.code in REMOVE_IMAGE_CODES else issue.code)
        if key in done:
            continue
        done.add(key)
        record = _fix_image_level(ds, issue, cfg)
        if record is not None:
            fixes.append(record)

    for stem, codes_by_line in line_codes.items():   # sau khi xoá ảnh: file đã xoá sẽ được bỏ qua êm
        fixes.extend(fix_label_file(ds, stem, codes_by_line))
    return fixes
