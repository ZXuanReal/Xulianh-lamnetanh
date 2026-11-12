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

---



## 8. Định Hướng Phát Triển



Dự án tiếp tục được hoàn thiện nhằm nâng cao trải nghiệm người dùng và chất lượng xử lý ảnh.



- Cải thiện giao diện và tối ưu quy trình thao tác trên ứng dụng.

- Mở rộng kiểm thử với nhiều định dạng và kích thước ảnh.

- Tối ưu hiệu năng xử lý ảnh trên CPU/GPU.

- Tiếp tục bảo trì mã nguồn và hoàn thiện tài liệu hướng dẫn.



---

---




## 8. ĐỊNH HƯỚNG PHÁT TRIỂN

Trong tương lai, dự án sẽ tiếp tục được nghiên cứu, hoàn thiện và mở rộng nhằm nâng cao chất lượng xử lý ảnh, tối ưu hiệu năng hoạt động cũng như cải thiện trải nghiệm người dùng. Mặc dù hệ thống hiện tại đã xây dựng được những chức năng xử lý ảnh cơ bản và đáp ứng các yêu cầu chính của đề tài, vẫn còn nhiều tiềm năng để phát triển thêm về mặt kỹ thuật, khả năng tương thích và tính ứng dụng thực tế. Các định hướng phát triển được đề xuất như sau:

### 8.1. Cải thiện giao diện và nâng cao trải nghiệm người dùng

Tiếp tục hoàn thiện giao diện ứng dụng theo hướng hiện đại, trực quan và thân thiện hơn với người sử dụng. Bố cục các chức năng cần được sắp xếp khoa học, giúp người dùng dễ dàng lựa chọn thao tác và theo dõi quá trình xử lý ảnh mà không cần có kiến thức chuyên sâu về kỹ thuật.

Bên cạnh đó, có thể bổ sung các tính năng hỗ trợ như:
- Xây dựng khu vực xem trước hình ảnh trước và sau khi xử lý để người dùng dễ dàng so sánh chất lượng.
- Bổ sung thanh tiến trình (Progress Bar) nhằm hiển thị trạng thái và thời gian xử lý ảnh.
- Cho phép người dùng tùy chỉnh các thông số xử lý như mức độ làm nét, độ sáng, độ tương phản và cường độ khử nhiễu.
- Bổ sung chức năng xem lại lịch sử xử lý và quản lý các hình ảnh đã xuất.
- Tối ưu quy trình tải ảnh, lựa chọn chức năng, xử lý và tải kết quả nhằm giảm số lượng thao tác không cần thiết.

### 8.2. Mở rộng khả năng hỗ trợ định dạng và kích thước ảnh

Một trong những hướng phát triển quan trọng là mở rộng khả năng tương thích với nhiều loại dữ liệu hình ảnh khác nhau. Hệ thống cần được kiểm thử trên nhiều định dạng phổ biến như JPG, JPEG, PNG, BMP, TIFF và WebP để đảm bảo khả năng đọc, xử lý và xuất ảnh ổn định.

Ngoài ra, cần mở rộng phạm vi kiểm thử với các hình ảnh có độ phân giải và kích thước khác nhau, từ ảnh có dung lượng nhỏ đến ảnh chất lượng cao, ảnh chụp từ điện thoại và ảnh được lưu trữ từ các thiết bị chuyên dụng.

Đặc biệt, cần nghiên cứu phương pháp xử lý các hình ảnh có kích thước lớn nhằm hạn chế tình trạng tiêu tốn quá nhiều bộ nhớ RAM hoặc gây gián đoạn trong quá trình thực thi. Có thể áp dụng kỹ thuật chia ảnh thành các vùng nhỏ (Image Tiling) để xử lý từng phần, sau đó ghép lại thành hình ảnh hoàn chỉnh mà vẫn đảm bảo chất lượng đầu ra.

### 8.3. Tối ưu hiệu năng xử lý ảnh trên CPU và GPU

Hiệu năng xử lý là yếu tố quan trọng ảnh hưởng trực tiếp đến khả năng ứng dụng thực tế của hệ thống. Vì vậy, dự án cần tiếp tục nghiên cứu và tối ưu các thuật toán nhằm giảm thời gian thực thi, hạn chế tài nguyên tiêu thụ và nâng cao tốc độ phản hồi.

Đối với CPU, có thể cải thiện hiệu năng thông qua việc tối ưu mã nguồn, giảm các thao tác tính toán dư thừa, tận dụng xử lý đa luồng (Multi-threading) và tối ưu quy trình quản lý bộ nhớ.

Đối với GPU, có thể nghiên cứu tích hợp các công nghệ tăng tốc như CUDA hoặc những thư viện hỗ trợ tính toán song song, giúp tăng tốc các tác vụ xử lý ảnh có khối lượng tính toán lớn. Việc lựa chọn CPU hay GPU cần được thực hiện linh hoạt dựa trên cấu hình phần cứng và quy mô hình ảnh đầu vào.

Ngoài ra, cần tiến hành đo lường và so sánh thời gian xử lý trên các môi trường phần cứng khác nhau để đánh giá chính xác hiệu quả của những phương pháp tối ưu.

### 8.4. Nâng cao chất lượng và độ chính xác xử lý ảnh

Dự án có thể tiếp tục phát triển các phương pháp xử lý ảnh chuyên sâu nhằm cải thiện chất lượng hình ảnh đầu ra, đặc biệt đối với những hình ảnh bị mờ, nhiễu, thiếu sáng hoặc có độ phân giải thấp.

Một số hướng nghiên cứu có thể triển khai:
- Nghiên cứu và tích hợp các thuật toán khử nhiễu nâng cao để giảm hiện tượng nhiễu hạt nhưng vẫn giữ được chi tiết hình ảnh.
- Cải thiện khả năng làm sắc nét đường biên và các chi tiết nhỏ, hạn chế hiện tượng ảnh bị gắt hoặc xuất hiện nhiễu giả.
- Nghiên cứu các phương pháp tăng độ phân giải ảnh (Image Super-Resolution) bằng mô hình học sâu.
- Kết hợp các kỹ thuật cân bằng sáng, điều chỉnh màu sắc và tăng cường độ tương phản để cải thiện chất lượng hình ảnh trong nhiều điều kiện khác nhau.
- Đánh giá chất lượng ảnh đầu ra bằng các chỉ số khách quan như PSNR, SSIM và so sánh trực quan giữa ảnh gốc với ảnh sau xử lý.

Định hướng này giúp hệ thống không chỉ thực hiện được các thao tác xử lý ảnh cơ bản mà còn có khả năng giải quyết những trường hợp hình ảnh phức tạp hơn.

### 8.5. Mở rộng chức năng và khả năng tích hợp hệ thống

Trong các phiên bản tiếp theo, có thể xem xét phát triển thêm những chức năng hỗ trợ người dùng trong quá trình xử lý và quản lý hình ảnh, chẳng hạn như xử lý hàng loạt nhiều ảnh cùng lúc (Batch Processing), tự động nhận diện đặc điểm hình ảnh để đề xuất thông số xử lý phù hợp và cho phép lưu lại các cấu hình xử lý thường xuyên sử dụng.

Bên cạnh đó, hệ thống có thể được nghiên cứu để tích hợp với các nền tảng lưu trữ trực tuyến hoặc xây dựng giao diện lập trình ứng dụng (API), tạo điều kiện cho các ứng dụng khác có thể sử dụng chức năng xử lý ảnh mà không cần triển khai lại toàn bộ thuật toán.

Về lâu dài, có thể phát triển ứng dụng theo mô hình Client – Server hoặc triển khai trên nền tảng Web, giúp người dùng truy cập và sử dụng các chức năng xử lý ảnh từ nhiều thiết bị khác nhau mà không cần cài đặt phần mềm trực tiếp trên máy tính.

### 8.6. Tăng cường kiểm thử, bảo trì và hoàn thiện tài liệu

Để đảm bảo tính ổn định và khả năng phát triển lâu dài, dự án cần được kiểm thử thường xuyên trên nhiều môi trường hệ điều hành, cấu hình phần cứng và điều kiện sử dụng khác nhau.

Cần xây dựng thêm các bộ kiểm thử (Test Cases) cho từng chức năng, bao gồm trường hợp đầu vào hợp lệ, dữ liệu không hợp lệ, hình ảnh bị lỗi, định dạng không được hỗ trợ và các tình huống phát sinh trong quá trình xử lý. Đồng thời, cần kiểm tra khả năng phục hồi khi xảy ra lỗi để hạn chế tình trạng ứng dụng bị dừng đột ngột hoặc mất dữ liệu.

Về mặt mã nguồn, cần tiếp tục thực hiện bảo trì, tái cấu trúc những đoạn code chưa tối ưu, chuẩn hóa cách đặt tên biến, hàm và tổ chức các module theo hướng dễ quản lý, dễ mở rộng. Việc quản lý phiên bản bằng Git cũng cần được duy trì nhằm theo dõi lịch sử thay đổi và hỗ trợ quá trình phối hợp phát triển.

Song song đó, tài liệu hướng dẫn cài đặt, cấu hình môi trường, sử dụng chức năng và mô tả kiến trúc hệ thống cần được cập nhật đầy đủ. Điều này giúp người dùng mới có thể tiếp cận ứng dụng thuận tiện hơn, đồng thời hỗ trợ những người phát triển trong việc tiếp tục nâng cấp và bảo trì dự án.

