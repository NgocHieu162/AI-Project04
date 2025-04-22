"""
Air Quality Prediction Model - Training Script

Script này thực hiện việc huấn luyện mô hình dự đoán chất lượng không khí
theo hai phương pháp: tuyến tính cơ bản và XGBoost nâng cao.

Cách sử dụng:
    python train.py --data train.csv --model linear  # Để huấn luyện mô hình tuyến tính
    python train.py --data train.csv --model advanced  # Để huấn luyện mô hình XGBoost nâng cao
    python train.py --data train.csv --model both  # Để huấn luyện cả hai mô hình

Tác giả: [Tên của bạn]
Ngày: 21/04/2025
"""

import argparse
import os
import time
from linear_model import run_linear_model
from advanced_model import run_advanced_model

def parse_arguments():
    """
    Phân tích các đối số dòng lệnh.
    
    Trả về:
        argparse.Namespace: Các đối số được phân tích
    """
    parser = argparse.ArgumentParser(description='Train air quality prediction models.')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Đường dẫn đến file dữ liệu CSV')
    
    parser.add_argument('--model', type=str, default='both', choices=['linear', 'advanced', 'both'],
                        help='Loại mô hình để huấn luyện: linear, advanced, hoặc both')
    
    return parser.parse_args()


def main():
    """
    Hàm chính để chạy quá trình huấn luyện.
    """
    # Phân tích đối số dòng lệnh
    args = parse_arguments()
    
    # Kiểm tra file dữ liệu tồn tại
    if not os.path.exists(args.data):
        print(f"Lỗi: File dữ liệu '{args.data}' không tồn tại!")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Đã tạo thư mục đầu ra: models")
    
    # Bắt đầu huấn luyện
    start_time = time.time()
    
    if args.model in ['linear', 'both']:
        print("\n" + "="*50)
        print("HUẤN LUYỆN MÔ HÌNH TUYẾN TÍNH")
        print("="*50)
        run_linear_model(args.data)
    
    if args.model in ['advanced', 'both']:
        print("\n" + "="*50)
        print("HUẤN LUYỆN MÔ HÌNH NÂNG CAO (XGBoost)")
        print("="*50)
        run_advanced_model(args.data)
    
    # Hiển thị thời gian chạy
    elapsed_time = time.time() - start_time
    print(f"\nTổng thời gian huấn luyện: {elapsed_time:.2f} giây")
    print(f"Các mô hình đã được lưu trong thư mục: models")
    print("\nHoàn thành!")

if __name__ == "__main__":
    main()