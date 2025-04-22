"""
Air Quality Prediction Model - Prediction Script

Script này sử dụng các mô hình đã huấn luyện để dự đoán nồng độ Ozone và NO2
từ dữ liệu cảm biến mới.

Cách sử dụng:
    python predict.py --data new_data.csv --model linear  # Sử dụng mô hình tuyến tính
    python predict.py --data new_data.csv --model advanced  # Sử dụng mô hình XGBoost nâng cao
    python predict.py --data new_data.csv --output predictions.csv  # Chỉ định file đầu ra

Tác giả: [Tên của bạn]
Ngày: 21/04/2025
"""

import argparse
import os
import joblib
from utils import load_data, preprocess_data, engineer_features_advanced, get_linear_features, get_advanced_features

def parse_arguments():
    """
    Phân tích các đối số dòng lệnh.
    
    Trả về:
        argparse.Namespace: Các đối số được phân tích
    """
    parser = argparse.ArgumentParser(description='Predict air quality using trained models.')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Đường dẫn đến file dữ liệu CSV cần dự đoán')
    
    parser.add_argument('--model', type=str, default='advanced', choices=['linear', 'advanced'],
                        help='Loại mô hình để sử dụng: linear hoặc advanced')
    
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Tên file CSV để lưu kết quả dự đoán')
    
    return parser.parse_args()


def load_models(model_type):
    """
    Tải các mô hình đã huấn luyện.
    
    Tham số:
        model_type (str): Loại mô hình ('linear' hoặc 'advanced')
        
    Trả về:
        tuple: (mô hình Ozone, mô hình NO2)
    """
    if model_type == 'linear':
        ozone_model_path = 'models/model_Ozone_Task1.pkl'
        no2_model_path = 'models/model_NO2_Task1.pkl'
    else:  # advanced
        ozone_model_path = 'models/model_Ozone_Task2.pkl'
        no2_model_path = 'models/model_NO2_Task2.pkl'
    
    # Kiểm tra xem các file mô hình có tồn tại không
    if not os.path.exists(ozone_model_path) or not os.path.exists(no2_model_path):
        raise FileNotFoundError(f"Không tìm thấy các file mô hình {model_type}. "
                               f"Vui lòng chạy train.py trước.")
    
    # Tải mô hình
    ozone_model = joblib.load(ozone_model_path)
    no2_model = joblib.load(no2_model_path)
    
    return ozone_model, no2_model


def predict_air_quality(data_path, model_type, output_path):
    """
    Dự đoán chất lượng không khí từ dữ liệu mới.
    
    Tham số:
        data_path (str): Đường dẫn đến file dữ liệu mới
        model_type (str): Loại mô hình ('linear' hoặc 'advanced')
        output_path (str): Đường dẫn để lưu kết quả dự đoán
    """
    # Tải mô hình
    ozone_model, no2_model = load_models(model_type)
    
    # Tải và xử lý dữ liệu
    df = load_data(data_path)
    
    if model_type == 'linear':
        # Chỉ lấy các đặc trưng cơ bản cho mô hình tuyến tính
        X = get_linear_features(df)
    else:  # advanced
        # Tiền xử lý và tạo đặc trưng nâng cao cho mô hình XGBoost
        df = preprocess_data(df)
        df = engineer_features_advanced(df)
        X = get_advanced_features(df)
    
    # Dự đoán
    df['Predicted_OZONE'] = ozone_model.predict(X)
    df['Predicted_NO2'] = no2_model.predict(X)
    
    # Lưu kết quả
    output_columns = ['Time', 'o3op1', 'o3op2', 'no2op1', 'no2op2', 'Predicted_OZONE', 'Predicted_NO2']
    if 'OZONE' in df.columns and 'NO2' in df.columns:
        output_columns.extend(['OZONE', 'NO2'])
        # Tính sai số nếu có giá trị thực
        df['OZONE_Error'] = abs(df['OZONE'] - df['Predicted_OZONE'])
        df['NO2_Error'] = abs(df['NO2'] - df['Predicted_NO2'])
        output_columns.extend(['OZONE_Error', 'NO2_Error'])
    
    df[output_columns].to_csv(output_path, index=False)
    print(f"Kết quả dự đoán đã được lưu vào {output_path}")


def main():
    """
    Hàm chính để chạy quá trình dự đoán.
    """
    # Phân tích đối số dòng lệnh
    args = parse_arguments()
    
    # Kiểm tra file dữ liệu tồn tại
    if not os.path.exists(args.data):
        print(f"Lỗi: File dữ liệu '{args.data}' không tồn tại!")
        return
    
    # Thực hiện dự đoán
    try:
        predict_air_quality(args.data, args.model, args.output)
        print("\nHoàn thành dự đoán!")
    except Exception as e:
        print(f"Lỗi khi dự đoán: {str(e)}")


if __name__ == "__main__":
    main()