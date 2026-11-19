# Báo cáo Day 09 — Phan Đình Giáp, 2A202603252

## 1. Kết luận trước
**HOLD** (mã thoát `python -m pipeline.run` = **10**).
- **Vào:** 224 ảnh. **Ra:** 202 ảnh trong `export/clean_v1`. Bước 1 loại 22 ảnh: hỏng, nhỏ, trùng, rò rỉ test.
- **Lỗi bắt buộc còn lại:** 0 (`08_summary.json`: `n_errors = 0`).
- **Lý do Hold:** 38/202 ảnh (**18.8%**) còn cảnh báo ngữ nghĩa chưa được người duyệt, vượt ngưỡng 10%. Hàng đợi có 125 ảnh nhưng ngân sách duyệt chỉ 80 ảnh.
- Bản export hiện tại là **bản nháp**, chưa được phát hành.

## 2. Lỗi và phân việc
Số liệu lấy từ `01_intake.csv` và `02_error_list_v1.csv`: 211 issue trên 149 ảnh. Bản sửa của máy lấy từ `03_fixes.csv` (35 thay đổi).

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT | 8 ảnh | máy (loại ảnh) | md5 trùng từng byte nên chắc chắn là trùng. Giữ ảnh đầu tiên. |
| DUP_IMAGE_NEAR | 4 ảnh | máy (loại ảnh) | phash ≤ 10. Mình đã mở xem, đều là trùng thật (xem §3). |
| LEAK_TEST | 5 ảnh | máy (loại ảnh) | Trùng với ảnh trong `test/`. Giữ lại thì mAP trên test bị thổi phồng. |
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 ảnh | máy (loại ảnh) | Cạnh ngắn < 250px, hoặc ảnh không `load()` được. |
| BAD_CLASS_ID | 6 dòng | máy (`drop_line`) | Id 93, 4, 7, −1 không thuộc lớp nào. Đoán lại lớp là bịa nhãn, nên xoá dòng. |
| BAD_FORMAT | 3 dòng | máy (`drop_line`) | Thiếu cột hoặc có `nan`, không có box để giữ. |
| BOX_ZERO | 4 dòng | máy (`drop_line`) | Box rộng hoặc cao bằng 0. |
| DUP_BOX | 7 dòng | máy (`drop_line`) | Cùng lớp, IoU > 0.9 với dòng trước (ví dụ `img_0001` dòng 3, `img_0006` dòng 2). |
| BOX_OUT_OF_IMAGE | 8 dòng | máy (`clip_box`) | Cắt box về [0,1], giữ lớp (ví dụ `img_0030` dòng 9, tâm x = −0.0038). |
| BAD_CLASS_NAME | 1 file (2 tên) | máy (`rename_classes`) | `data.yaml` ghi `Person` và `Bus` viết hoa. Ghi lại theo `classes`. |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | Máy không được bịa nhãn. File rỗng nghĩa là "ảnh nền", khác với "chưa gán". |
| POSSIBLE_MISSING | 53 / 52 ảnh | **người** | yolo26l thấy vật mà nhãn không có (ví dụ người lính trong `img_0001`). |
| POSSIBLE_LOOSE_BOX | 28 / 24 ảnh | **người** | Đúng lớp nhưng IoU < 0.5. Box nào mới đúng thì cần mắt người. |
| POSSIBLE_FALSE_POS | 29 / 27 ảnh | **người** | Box nhãn không khớp vật nào model thấy. Model cũng có thể sót. |
| POSSIBLE_WRONG_CLASS | 18 ảnh | **người** | Lớp nhãn khác lớp model (car/truck/bus). Cả hai đều có thể sai. |
| EMPTY_LABEL | 12 ảnh | **người** | File rỗng: ảnh nền hay bị thiếu nhãn? (ví dụ `img_0004` có xe con nhưng file rỗng). |
| BOX_TINY / BOX_HUGE | 6 dòng / 3 dòng | **người** | Chỉ là cảnh báo theo guideline. Máy không có căn cứ để xoá. |
| CLASS_RARE | 4 (cấp dataset) | máy gián tiếp | Lớp −1, 4, 7, 93: là hệ quả của BAD_CLASS_ID và hết sau `drop_line`. |

**Đầu hàng đợi.** `05_review_queue.csv` xếp theo tổng trọng số `review.weights × (1 + conf)`.
- `NEEDS_LABEL` = 10, nên 6 ảnh thiếu nhãn đứng ngay đầu: chưa có người gán thì gate sẽ ra Rework.
- `img_0020` đứng thứ nhất (11.52) vì cộng cả FIXED_DROP_LINE (2), POSSIBLE_WRONG_CLASS (3) và POSSIBLE_MISSING (2), đều nhân theo conf của model.

**Người xem được bao nhiêu lỗi ngữ nghĩa.**
- Sau autofix, `04_error_list_v2` còn **123 cảnh báo ngữ nghĩa** (`POSSIBLE_*`) trên 103 ảnh.
- Người duyệt 80/125 ảnh trong hàng đợi (`06_review_log.csv`). Số ảnh đó phủ **94/123 cảnh báo (76%) trên 74/103 ảnh**.
- Còn 45 ảnh trong hàng đợi không ai xem.

## 3. Calibration (2 ngưỡng)
Calibrate trên 39 ảnh của `data/calib`, có đáp án `MANIFEST.csv`. Bảng sweep đầy đủ nằm ở `submission/calibration/calibration_report.md`.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **10** | DUP_IMAGE_NEAR: 1.00 / 0.50 → 1.00 / 0.67 (EXACT giữ 1.00 / 1.00) | Đây là lỗi bị **tự xoá ảnh**, nên ưu tiên precision. Từ 4 đến 14, calib cho cùng một kết quả (1.00 / 0.67). Chọn 10 vì trên raw, các cặp trùng thật nằm ở khoảng cách 0–10 (`img_0061`/`img_0186` = 10), còn cặp ảnh khác nhau gần nhất nằm ở 16. Không lấy 16 dù F1 calib cao nhất (0.91), xem đoạn dưới. Ở 20, cả EXACT lẫn NEAR đều tụt (0.33 / 0.50) vì gộp nhầm ảnh khác nhau. |
| `second_opinion.conf` | 0.25 | **0.6** | MISSING 0.50 / 1.00 → 0.57 / 1.00 · WRONG_CLASS 0.56 / 0.83 → 0.62 / 0.83 · LOOSE_BOX 0.29 / 0.40 → 0.20 / 0.20 · FALSE_POS 0.31 / 1.00 (không đổi, dùng `fp_conf`) | Đây chỉ là **cảnh báo cho người**, nên giữ recall. 0.6 là mức conf cao nhất mà MISSING vẫn có recall 1.00 và WRONG_CLASS vẫn 0.83, đồng thời precision tăng (bớt báo động giả). Trên raw, hàng đợi duyệt giảm từ 143 xuống 125 ảnh. Lên 0.7 thì MISSING tụt recall còn 0.88, lên 0.8 còn 0.75. Đánh đổi chấp nhận: LOOSE_BOX tụt recall (0.40 → 0.20), nhưng chỉ có 5 lỗi chèn, tức hơn kém đúng 1 lỗi. |

**Một chỗ calibration có thể đánh lừa: `phash_dist = 16` đẹp nhất trên calib nhưng xoá nhầm ảnh trên raw.**
- **Trên calib:** 16 cho DUP_IMAGE_NEAR = 1.00 / 0.83 / F1 0.91, cao nhất bảng, precision vẫn 1.00. Lý do là trên calib, cặp ảnh khác nhau gần nhất cách 20, còn cặp trùng `calib_015`/`calib_037` nằm ở 16.
- **Trên raw (đo phash từng cặp, raw + test + gold_val, rồi mở xem):**
  - Ở khoảng cách **16** có 9 cặp. Nhìn bằng mắt thì **8 cặp là ảnh khác hẳn nhau**: `img_0008` (trượt tuyết) với `img_0011` (tennis), `img_0013` (bàn máy tính) với `test_087` (bóng chày), `img_0017` với `test_071`, `img_0021` với `test_048`, `img_0033` với `test_070`, `img_0038` với `test_002`, `img_0113` với `img_0177`, `img_0115` với `test_088`.
  - Chỉ có `img_0074`/`img_0084` (bản bị cắt) là trùng thật.
  - Khoảng cách 18 đã có 51 cặp.
- **Hậu quả trên raw:** chạy pipeline với 16 thì bước 1 nhận 193/224 ảnh, so với 202/224 khi dùng 10. Trong 9 ảnh bị loại thêm, chỉ 1 ảnh là trùng thật; khoảng 8 ảnh tốt bị xoá, một phần bị gắn nhầm là `LEAK_TEST`.
- **Nguyên nhân:** calib quá nhỏ (39 ảnh, chỉ 6 ca gần trùng) nên không thấy được chỗ chồng lấn giữa "trùng bị biến đổi mạnh" và "ảnh khác nhưng bố cục giống" trên tập lớn hơn.
- **Hệ quả:** ở bất kỳ ngưỡng nào cũng còn sót. Với 10, pipeline bỏ sót `calib_037` (16), `calib_038` (18) và `img_0074`/`img_0084` (16) trên raw. Đây là **điểm mù đã biết**: phash không tách được ảnh bị cắt mạnh khỏi ảnh khác có bố cục giống nhau.
- **Kết quả với 10:** bước 1 loại 22 ảnh: 8 trùng byte, 4 gần trùng, 5 rò rỉ test, 3 sai kích thước, 2 ảnh hỏng. Mình đã mở xem các cặp ở khoảng cách 2, 6 và 10, đều là trùng thật.

*Không chọn `conf = 0.8` dù nó cho Release* (7.9% ảnh còn cảnh báo, dưới ngưỡng 10%). Lý do: MISSING tụt recall xuống 0.75, tức hàng đợi nhỏ đi vì checker mù hơn chứ không phải vì dữ liệu sạch hơn. Như vậy là Release giả.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Lệnh chạy: `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt).
- Log in ra: `kết luận: REWORK — còn 6 lỗi bắt buộc (ERROR): phải sửa trước khi xuất; 112/202 ảnh còn cảnh báo chưa duyệt (55.5%)`.
- **Mã thoát 20.**
- Bước 5 in `KHÔNG xuất: còn lỗi bắt buộc (Rework) — trả về bước 3`, và thư mục `work/exp_nb` **không được tạo**.
- 6 lỗi đó chính là 6 ảnh `MISSING_LABEL_FILE` (`img_0031`, `0053`, `0126`, `0135`, `0170`, `0219`). Autofix chỉ ghi `needs_label`, không tạo file nhãn, nên lỗi vẫn còn đến khi có người gán.
- Khi có người duyệt 80 ảnh, 6 ảnh này đứng đầu hàng đợi và được gán nhãn. Lỗi bắt buộc về 0 và kết luận chuyển sang Hold (mã 10), không phải Release, vì vẫn còn 18.8% ảnh có cảnh báo chưa duyệt.

**(b) Điểm mù có bằng chứng.**
1. **Ảnh trùng bị biến đổi mạnh lọt qua.** `img_0074` và `img_0084` là cùng một cảnh (bản bị cắt), phash cách nhau 16 > ngưỡng 10. Cả hai đều không có trong `01_intake.csv` và đều vào `export/clean_v1`. Trên calib, `calib_037` (16) và `calib_038` (18) cũng bị sót, nên DUP_IMAGE_NEAR recall chỉ 0.67. Nâng ngưỡng lên 16 thì xoá nhầm khoảng 8 ảnh tốt (§3), nên không có ngưỡng phash nào vừa bắt được hết vừa an toàn.
2. **Box lỏng nhẹ gần như không bị phát hiện.** Trên calib, POSSIBLE_LOOSE_BOX chỉ đạt **precision 0.20 / recall 0.20** (bắt 1/5 lỗi chèn). Box phình nhẹ vẫn có IoU ≥ `iou_loose = 0.5` (ngưỡng cố định) với dự đoán của model, nên không bị cờ.
3. **Nhầm lớp mà model lớn cũng nhầm, và checker báo động giả.** Calib: WRONG_CLASS recall 0.83, sót 1/6 lỗi chèn. Lỗi thật trong raw lại nghiêng về car/truck/bus, là nhóm model dễ lẫn hơn person→car trong calib. Ngoài ra, sau khi người thay bằng nhãn chuẩn, `07_error_list_v3` vẫn còn **25/80 ảnh đã duyệt bị cờ** (11 MISSING, 10 FALSE_POS, 4 LOOSE, 3 WRONG_CLASS). Đó là chỗ model và người bất đồng, tức cảnh báo của máy không thay được mắt người.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Cấu hình.** Chạy trên Colab T4 bằng `notebooks/day09_train_compare.ipynb`.
- Model `yolo26n.pt`, 20 epoch, batch 16, ảnh 640, **2 seed (0, 1)** cho mỗi biến thể.
- Cùng tập val (`gold_val`, 40 ảnh) và cùng tập **test** (100 ảnh, nhãn người).
- `clean` = `export/clean_v1`: 202 ảnh, `manifest_sha1 = 4278b929…`, khớp với bài nộp. `raw` = 224 ảnh gốc, không sửa gì.

Bảng sau lấy từ `submission/results/comparison.md`:

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 202 | 2 | 0.701 ± 0.025 | 0.487 ± 0.008 | 0.759 ± 0.003 | 0.625 ± 0.019 |

**So với nhiễu.**
- Chênh mAP50 (clean − raw) = **−0.025**. Con số này **bằng đúng độ lệch giữa 2 seed của clean** (0.025): mAP50 của clean là 0.718 ở seed 0 và 0.683 ở seed 1, tức hai seed chênh nhau 0.035.
- Seed 0 của clean (0.718) chỉ kém raw 0.008.
- Với 2 seed, ước lượng độ lệch rất thô. Độ lệch ≈ 0 của raw (0.7256 và 0.7259) nhiều khả năng là trùng hợp, không phải raw thật sự ổn định.

**Kết luận: chưa kết luận được.** Clean không tốt hơn raw trên tập test này, và mức kém hơn cũng nằm trong nhiễu. Kết quả không ủng hộ câu "làm sạch giúp model tốt hơn", nhưng cũng chưa chứng minh được là làm sạch gây hại.

**Theo lớp** (AP50 của seed 0 / seed 1):
- `truck` giảm rõ nhất: raw 0.657 / 0.640, clean 0.590 / 0.486.
- `car` (≈ 0.74), `person` (≈ 0.82) và `bus` (0.68–0.73) gần như không đổi.
- Phần chênh lệch chủ yếu đến từ `truck`, cũng là lớp dao động nhiều nhất giữa 2 seed của clean (0.590 so với 0.486).

**Giả thuyết và cách kiểm:**
1. **Rò rỉ test làm điểm raw cao hơn thật.**
   - `raw` chứa 5 ảnh trùng tập test mà pipeline đã báo `LEAK_TEST`: `img_0049`↔`test_033`, `img_0174`↔`test_085`, `img_0221`↔`test_009` (trùng byte); `img_0167`↔`test_044`; `img_0101`↔`test_039`. Clean đã loại 5 ảnh này.
   - Cách kiểm: đánh giá lại cả 4 model trên test **bỏ 5 ảnh trên** (95 ảnh). Nếu khoảng cách thu hẹp thì một phần lợi thế của raw là do rò rỉ.
2. **`truck` quá ít mẫu, nhiễu lớn.**
   - Toàn lô chỉ có 41 box truck (`class_stats.csv`), so với 574 box person. Clean còn mất thêm ảnh vì loại trùng và xoá dòng, mỗi ảnh có truck bị bớt đều làm AP truck đổi mạnh.
   - Cách kiểm: đếm box truck trong `work/train/{raw,clean}` (ô 5 của notebook), chạy thêm seed (≥ 5) và xem độ lệch AP truck.
3. **Clean chưa thật sạch, và nhãn không đồng nhất.**
   - Gate ra Hold: 38/202 ảnh còn cảnh báo chưa duyệt, LOOSE_BOX gần như không bắt được (recall 0.20, xem §3).
   - 80 ảnh được thay bằng nhãn người, các ảnh còn lại vẫn là nhãn máy, nên hai phong cách nhãn bị trộn lẫn.
   - Cách kiểm: train biến thể `oracle` (nhãn người, cùng 202 ảnh) làm chặn trên. Nếu oracle cũng không vượt raw thì vấn đề nằm ở cỡ dữ liệu hoặc độ nhiễu, không phải ở nhãn.
4. **Lỗi cú pháp ít ảnh hưởng tới train.** Ultralytics vốn tự bỏ qua dòng hoặc ảnh hỏng của raw (id lớp lạ, box tràn), nên phần autofix sửa có thể không tạo khác biệt. Ngược lại, clean ít hơn raw 22 ảnh (−10%).
   - Cách kiểm: đọc cảnh báo "ignoring corrupt image/label" ở ô 6 của notebook để biết raw thực sự train trên bao nhiêu ảnh.

**Việc tiếp theo:**
- Chạy ≥ 5 seed.
- Đánh giá trên test đã bỏ 5 ảnh rò rỉ.
- Thêm biến thể `oracle`.
- Người duyệt thêm 45 ảnh còn trong hàng đợi, rồi train lại.

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối: HOLD.** Không phát hành `clean_v1`. Đây là bản nháp, dùng được cho thí nghiệm.

Để chuyển thành Release mà **không** nới ngưỡng cố định:
- Tỉ lệ ảnh còn cảnh báo chưa duyệt phải < 10%, tức ≤ 20/202 ảnh. Hiện có 38 ảnh, nên cần **ít nhất 18 ảnh nữa được người duyệt**, cho một vòng duyệt thứ hai với 45 ảnh còn lại trong hàng đợi.
- Không nâng `second_opinion.conf` lên 0.8 để ép ra Release, vì recall của MISSING sẽ tụt xuống 0.75 (§3).
- Về lâu dài, cần cải thiện precision của FALSE_POS (hiện 0.31) và LOOSE_BOX (0.20), để hàng đợi bớt báo động giả và 80 ảnh duyệt được dùng vào lỗi thật.

**(b) Ghi chú dùng AI (Claude Code).**
- **Prompt chính:** "Làm Pha 2: viết 4 module (TDD), code đúng quy tắc kiểu dữ liệu và clean code Python". Kèm theo file hướng dẫn lab. AI đọc docstring stub, file test và `common.py` trước khi viết.
- **Cách kiểm:**
  - Chạy `pytest` từng file, rồi toàn bộ (**79 passed**).
  - Tự chạy thêm 3 ca khó: một dòng vừa tràn vừa lặp chỉ ra 1 `drop_line`; dòng trống không làm lệch số dòng; ảnh có nhiều dự đoán ra nhiều issue đúng thứ tự.
  - Chạy autofix lần 2 để kiểm idempotent.
  - Đối chiếu calibration trên lỗi chèn, rồi mở xem ảnh thật bằng `viz_labels.py` và các cặp ảnh bị cho là trùng.
  - Kiểm sha1 các file viết sẵn: không file nào bị sửa.
- **Một lần AI sai:** khi phân tích các cặp phash = 16, AI báo "7/8 cặp là ảnh khác nhau" trong khi thực tế có **9 cặp, 8 cặp khác nhau** (đếm sót cặp `img_0115`/`test_088`). Lỗi được phát hiện khi đối chiếu lại ảnh ghép và đã sửa trong §3. Ngoài ra, bản đầu của `autofix.fix()` dùng một tập chung để chống trùng cho cả ảnh lẫn `data.yaml`, khó đọc, nên đã viết lại thành hàm `_fix_image_level` và chống trùng theo cặp (ảnh, nhóm sửa).

*(Thưởng +2, tùy chọn)* CVAT: không làm.
