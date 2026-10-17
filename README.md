# ML-XC: phiếm hàm trao đổi–tương quan DFT học bằng máy (QM9)

Pipeline end-to-end dùng QM9 làm nguồn hình học, PySCF để lấy mật độ PBE và nhãn CCSD(T)/cc-pVDZ, JAX + Flax để huấn luyện
hiệu chỉnh XC `ΔE_θ[n] = Σ w ρ f_θ(ρ, s)` cộng vào năng lượng PBE. Chạy trên Google Colab (mount Drive) hoặc cục bộ (VS Code/Jupyter).

## Cấu trúc
```
ml_xc_colab/
├── notebooks/01..06_*.ipynb
├── data/{raw, processed, checkpoints}
├── utils/{data_utils.py, model_utils.py, dft_utils.py}
├── requirements.txt
└── README.md
```
Cài đặt cục bộ: `pip install -r requirements.txt`. Colab: notebook tự `%pip install` thứ cần dùng; đặt thư mục dự án ở `MyDrive/ml_xc_colab/`.

## Quy trình
| # | Notebook | Việc làm | Thời gian (ghi chú) |
|---|---|---|---|
| 1 | 01_download_qm9 | Tải QM9 (Figshare) + `uncharacterized.txt`, giữ phân tử ≤5 nguyên tử nặng | vài phút (tải ~86 MB + đọc 133 885 file) |
| 2 | 02_filter_prepare | Chọn ngẫu nhiên 50 phân tử (seed 42), chia 30/10/10 theo phân tử, thêm 5 nguyên tử cô lập | < 1 phút |
| 3 | 03_run_dft_pbe | PBE/cc-pVDZ tự hợp, lưu ρ, ∇ρ, trọng số lưới | vài phút (đo: 5 nguyên tử + 7 phân tử nhỏ ≈ 40 s trên 1 core) |
| 4 | 04_run_ccsdt | CCSD(T)/cc-pVDZ toàn electron, dựng HDF5 | khoảng 1–4 giờ trên CPU (đo: C4H8, 96 hàm cơ sở ≈ 75 s trên 1 core; ngoại suy cho 5 nguyên tử nặng, chưa đo) |
| 5 | 05_train_ml_xc | Huấn luyện nhiều cấu hình, chọn theo validation | GPU nên dùng; cấu hình (128,128,64) chậm trên CPU |
| 6 | 06_benchmark | SCAN, HSE06 tự hợp + ML-XC, so với CCSD(T) | khoảng 30–60 phút (đo: 12 hệ nhỏ ≈ 3 phút) |

## Dữ liệu và nhãn
- Nguồn: QM9 (133 885 phân tử C/H/N/O/F, hình học B3LYP/6-31G(2df,p)). Loại 3 054 phân tử `uncharacterized` và phân tử số electron lẻ.
- Đầu vào: mật độ PBE tự hợp ρ(r) và |∇ρ| trên lưới tích phân PySCF (mức 3).
- Nhãn: **Δ = E_CCSD(T) − E_PBE[n_PBE]** cùng hình học và basis cc-pVDZ (nhãn là hiệu chỉnh năng lượng tổng, không phải E_xc trực tiếp).
- Nguyên tử cô lập H, C, N, O, F (UKS/UCCSD(T), spin cơ bản) luôn thuộc train, để tính năng lượng nguyên tử hóa.
- Định dạng: `data/processed/systems.csv` + `dataset_manifest.csv` (metadata), `qm9_xc_dataset.h5` (mật độ, năng lượng).

## Mô hình
- Flax MLP trên đặc trưng (log10 ρ, asinh s); lớp cuối khởi tạo bằng 0 nên mô hình ban đầu chính là PBE.
- Loss: MSE của Δ trên mỗi nguyên tử (kcal/mol) + phạt ràng buộc Lieb–Oxford.
- Lưới siêu tham số gồm cấu hình gốc (128,128,64; Adam lr 1e-4) và các mô hình nhỏ có weight decay; chọn theo RMSE năng lượng nguyên tử hóa trên validation. Test không được nạp ở bước 05.

## Benchmark và điều cần biết
- Thước đo chính: sai số **năng lượng nguyên tử hóa** so với CCSD(T)/cc-pVDZ (kcal/mol) với MAE, RMSE, max và khoảng tin cậy bootstrap 95%. Tổng năng lượng DFT lệch CCSD(T) hàng trăm mHa nên không dùng làm thước đo.
- **Nhãn bị giới hạn bởi basis**: CCSD(T)/cc-pVDZ thiếu năng lượng tương quan, nên năng lượng nguyên tử hóa của nó thấp hơn giá trị thật và PBE/SCAN/HSE06 đều "sai" hàng chục kcal/mol so với nhãn này. Một mô hình khớp được nhãn này chủ yếu học độ thiếu hụt basis, chưa có nghĩa là gần độ chính xác CCSD(T) ở giới hạn basis đầy đủ.
- ML-XC được đánh giá không tự hợp (PBE + ΔE_θ trên mật độ PBE). Chỉ có ~10 phân tử test, nên đọc kèm khoảng tin cậy.
