# ✨ Image Enhancement Studio

Ứng dụng Desktop chuyên nghiệp xây dựng bằng **Python + PySide6 + OpenCV + PyTorch**, kết hợp thuật toán xử lý tín hiệu ảnh truyền thống và mô hình học sâu (Real-ESRGAN & Face Restoration) để làm nét, khử nhiễu, cải thiện tương phản, tăng độ phân giải và phục hồi ảnh trên Windows.

---

## 1. Tính Năng Chính

1. **Làm nét ảnh chống viền sáng (Halo-Suppressed Sharpening)**:
   - **Unsharp Mask trên kênh độ sáng CIE LAB**: Giữ nguyên màu sắc gốc, tích hợp ngưỡng bảo vệ vùng phẳng (`Threshold`) để không làm nổi nhiễu và hàm nén mềm (`tanh` soft-knee clamping) hạn chế viền sáng Halo.
   - **Kernel Tích chập 3x3 đẳng hướng**: Tăng độ sắc cạnh nhanh với hệ số pha trộn mượt.
   - **Làm rõ Ảnh Tài liệu & Văn bản nhỏ**: Tự động cân bằng nền sáng không đều kết hợp làm nét biên chữ.
2. **Giảm nhiễu & Cải thiện Tương phản**:
   - Hỗ trợ **Non-Local Means (NLM)**, **Bilateral Filter** (tối ưu cho ảnh lớn > 6MP) và **Median Filter**.
   - **CLAHE trên không gian màu LAB**: Tăng tương phản cục bộ mà không làm lệch tông màu.
   - Điều chỉnh **Brightness** và **Gamma** qua bảng tra cứu LUT 256 mức tốc độ cao.
3. **Phục hồi & Siêu phân giải AI (Real-ESRGAN Native PyTorch)**:
   - Triển khai kiến trúc `SRVGGNetCompact` và `RRDBNet` thuần PyTorch tương thích 100% với trọng số chính thức của Real-ESRGAN (loại bỏ hoàn toàn xung đột thư viện `basicsr` cũ trên Python 3.12 / PyTorch 2.x / NumPy 2.x).
   - Hỗ trợ phóng to **2x** và **4x**, xử lý ảnh lớn an toàn bằng **Tiled Inference** (chia ô có đệm chồng lấp `tile_pad`), tái sử dụng mô hình trong RAM/VRAM và hỗ trợ hủy tác vụ giữa chừng.
   - **Phục hồi khuôn mặt tùy chọn (Face Restoration)**: Nhận diện khuôn mặt và tách tần số làm mịn da + tăng chi tiết mắt/môi với mặt nạ elip mờ biên.
4. **Chế độ Auto Enhance có giải thích**:
   - Tự động đo độ phân giải, chỉ số mờ (`Laplacian Variance`), độ sắc biên (`Tenengrad`), mức nhiễu (`MAD Noise Sigma`) và tương phản (`RMS Contrast`) để đề xuất phương pháp an toàn.
5. **Xử lý Hàng loạt (Batch Processing), Lịch sử SQLite & Bảo mật Cục bộ**:
   - Hỗ trợ `JPG`, `JPEG`, `PNG`, `WEBP` (bao gồm cả kênh trong suốt `RGBA` và đường dẫn tiếng Việt có dấu).
   - Không bao giờ ghi đè ảnh gốc; tự động đặt tên file đầu ra an toàn (`_enhanced`, `_1`, `_2`...).
   - Xử lý 100% cục bộ (Offline-First), không gửi ảnh hay metadata lên Internet.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
d:\Xử lí ảnh\
├── main.py                      # Điểm khởi chạy ứng dụng (hỗ trợ GUI và --smoke-test)
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── run_app.bat                  # Script khởi chạy nhanh trên Windows
├── build_exe.bat                # Script đóng gói EXE bằng PyInstaller
├── README.md                    # Tài liệu hướng dẫn cài đặt & sử dụng
├── config/
│   ├── __init__.py
│   └── settings.py              # Quản lý cấu hình lưu trữ JSON & đường dẫn tài nguyên
├── app/
│   ├── __init__.py
│   ├── main_window.py           # Cửa sổ chính (Sidebar, TopBar, Studio, Batch, History, Settings)
│   ├── styles.py                # Giao diện Modern Creative Studio (Dark Graphite QSS)
│   └── widgets/
│       ├── __init__.py
│       ├── image_viewer.py      # Khung vẽ ảnh phóng to, thu nhỏ, kéo thả (Drag & Drop)
│       ├── before_after_view.py # Thanh trượt Before/After & chế độ chia đôi đồng bộ
│       ├── settings_panel.py    # Bảng điều khiển thông số & trang Cài đặt hệ thống
│       └── progress_panel.py    # Thanh tiến trình, đồng hồ đo thời gian, cảnh báo & nhật ký
├── core/
│   ├── __init__.py
│   ├── image_io.py              # Đọc/ghi ảnh Unicode tiếng Việt, bảo toàn RGBA & EXIF/ICC
│   ├── image_analyzer.py        # Chẩn đoán ảnh, PSNR, SSIM, Sobel & Auto Enhance
│   ├── sharpen.py               # Unsharp Mask chống Halo, Kernel 3x3, Document Sharpen
│   ├── denoise.py               # NLM, Bilateral, Median, CLAHE & Brightness/Gamma LUT
│   ├── enhancement_pipeline.py  # Chuỗi xử lý kết hợp đa tầng & các Preset khởi đầu
│   └── batch_processor.py       # Xử lý hàng loạt cô lập lỗi từng file
├── ai/
│   ├── __init__.py
│   ├── device_manager.py        # Tự động phát hiện CPU/GPU CUDA & quản lý RAM/VRAM
│   ├── model_manager.py         # Quản lý danh mục trọng số AI, cache bộ nhớ & tải mô hình
│   ├── super_resolution.py      # Kiến trúc RRDBNet/SRVGGNetCompact & Tiled Inference
│   └── face_restoration.py      # Nhận diện & phục hồi khuôn mặt tùy chọn
├── workers/
│   ├── __init__.py
│   └── processing_worker.py     # QThread xử lý nền không làm treo giao diện
├── services/
│   ├── __init__.py
│   ├── history_service.py       # Lưu lịch sử xử lý cục bộ bằng SQLite
│   ├── export_service.py        # Dịch vụ xuất file an toàn
│   └── logging_service.py       # Ghi nhật ký xoay vòng (RotatingFileHandler) & bộ đệm UI
├── tests/
│   ├── __init__.py
│   ├── test_image_io.py         # Kiểm thử đọc/ghi file, tiếng Việt, RGBA, chống ghi đè
│   ├── test_sharpen.py          # Kiểm thử làm nét với PSNR, SSIM, chống Halo
│   ├── test_denoise.py          # Kiểm thử giảm nhiễu, CLAHE và Auto Enhance
│   └── test_batch_processor.py  # Kiểm thử xử lý lô, hủy tác vụ, AI Tiled Inference & History
├── assets/
│   └── icons/
│       └── studio_icon.svg
├── models/                      # Thư mục đặt các file trọng số AI (.pth)
├── outputs/                     # Thư mục xuất ảnh mặc định & cơ sở dữ liệu lịch sử
└── logs/                        # Thư mục lưu file log hoạt động
```

---

## 3. Hướng Dẫn Cài Đặt Trên Windows

### Bước 1: Tạo và kích hoạt môi trường ảo Python (Khuyến nghị Python 3.10 – 3.12)
Mở **PowerShell** tại thư mục dự án `d:\Xử lí ảnh`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 3: Cấu hình CPU hoặc GPU NVIDIA (CUDA)
- **Mặc định (CPU)**: Lệnh `pip install -r requirements.txt` cài đặt PyTorch chạy ổn định trên mọi máy tính Windows không cần card đồ họa rời.
- **Tăng tốc bằng GPU NVIDIA (CUDA 12.x)**: Nếu máy tính có GPU NVIDIA, cài bản PyTorch hỗ trợ CUDA:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
  Ứng dụng sẽ tự động nhận diện GPU khi chọn chế độ thiết bị là `Auto` hoặc `CUDA`. Nếu GPU không khả dụng, phần mềm tự động chuyển về `CPU` và hiển thị thông báo rõ ràng.

---

## 4. Hướng Dẫn Chạy Ứng Dụng & Kiểm Thử

- **Khởi chạy giao diện chính**:
  ```powershell
  python main.py
  ```
  hoặc nhấp đúp vào file `run_app.bat`.
- **Chạy kiểm thử giao diện tự động (Smoke Test)**:
  ```powershell
  python main.py --smoke-test
  ```
- **Chạy toàn bộ bộ kiểm thử tự động (Pytest)**:
  ```powershell
  python -m pytest -v
  ```

---

## 5. Hướng Dẫn Cài Đặt Trọng Số Mô Hình AI (.pth) & Giấy Phép

Ứng dụng **không tự ý tải file ngầm**. Bạn có thể tải trọng số trực tiếp trong giao diện (**Bảng điều khiển bên phải → Mục 4 → Nhấn nút "⬇ Tải trọng số AI đã chọn từ GitHub chính thức"**) hoặc tải thủ công và đặt vào thư mục `models/`:

| Tên mô hình | File trọng số cần đặt trong `models/` | Tỷ lệ | Dung lượng | Giấy phép | Nguồn phát hành chính thức |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Real-ESRGAN Compact v3** | `realesr-general-x4v3.pth` | 4x / 2x | ~4.7 MB | BSD-3-Clause | [xinntao/Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) |
| **Real-ESRGAN x2plus** | `RealESRGAN_x2plus.pth` | 2x | ~64.0 MB | BSD-3-Clause | [xinntao/Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| **Real-ESRGAN x4plus** | `RealESRGAN_x4plus.pth` | 4x | ~64.0 MB | BSD-3-Clause | [xinntao/Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| **Real-ESRGAN Anime 6B** | `RealESRGAN_x4plus_anime_6B.pth` | 4x | ~17.1 MB | BSD-3-Clause | [xinntao/Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) |

> **Giới hạn chất lượng & Cảnh báo trung thực**: Không có thuật toán hay mô hình AI nào có thể khôi phục chính xác 100% thông tin quang học đã mất hoàn toàn do rung nhòe nặng. Mô hình AI tái tạo chi tiết dựa trên học sâu và có thể sinh ra kết cấu hơi khác so với thực tế. Luôn sử dụng thanh trượt **Before/After** để đối chiếu trước khi lưu.

---

## 6. Hướng Dẫn Đóng Gói Thành File EXE Trên Windows

Để tránh làm file `.exe` phình to hàng trăm MB, hãy đóng gói theo chế độ thư mục (`--onedir`) và để thư mục `models/` nằm bên cạnh file `.exe`:

```powershell
pip install pyinstaller
.\build_exe.bat
```

Sau khi chạy xong, bản phân phối nằm tại `dist\ImageEnhancementStudio\ImageEnhancementStudio.exe`.

---

## 7. Khắc Phục Các Lỗi Thường Gặp (Troubleshooting)

1. **Thông báo "Chưa tìm thấy trọng số AI..."**:
   - *Nguyên nhân*: Bạn bật tùy chọn `Sử dụng mô hình AI Super-Resolution` nhưng chưa tải file `.pth` vào thư mục `models/`.
   - *Khắc phục*: Nhấn nút **"⬇ Tải mô hình..."** ngay trên bảng điều khiển để tải mô hình nhẹ `realesr-general-x4v3.pth` (~4.7 MB), hoặc bỏ chọn ô `Sử dụng mô hình AI Super-Resolution` để dùng bộ phóng to truyền thống `Edge-Directed Lanczos4`.
2. **Xử lý AI trên CPU chậm với ảnh lớn**:
   - *Nguyên nhân*: Mạng nơ-ron sâu thực hiện hàng tỷ phép tính tích chập; khi chạy trên CPU sẽ lâu hơn GPU.
   - *Khắc phục*: Chọn mô hình nhẹ `realesr-general-x4v3`, chọn tỷ lệ `2x`, hoặc giảm `Kích thước Tile` xuống `192 px` trong trang **Cài đặt**.
3. **Lỗi thiếu bộ nhớ (Out of Memory)**:
   - *Khắc phục*: Vào mục **Cài đặt**, giảm `Kích thước Tile` xuống `192 px` và nhấn **"🧹 Giải phóng bộ nhớ đệm AI (RAM/VRAM)"**.
