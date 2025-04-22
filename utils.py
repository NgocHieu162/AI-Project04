import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Các thông số cấu hình chung
TEST_SIZE = 0.2
RANDOM_STATE = 42

def save_model(model, filename):
    """
    Lưu mô hình vào file.
    
    Tham số:
        model: Mô hình cần lưu
        filename (str): Tên file để lưu mô hình
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV.
    
    Tham số:
        file_path (str): Đường dẫn đến file CSV
        
    Trả về:
        pd.DataFrame: DataFrame chứa dữ liệu
    """
    return pd.read_csv(file_path)

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá hiệu suất của mô hình.
    
    Tham số:
        model: Mô hình đã huấn luyện
        X_test (pd.DataFrame): Dữ liệu đặc trưng kiểm tra
        y_test (pd.Series): Biến mục tiêu kiểm tra
        
    Trả về:
        tuple: (RMSE, MAE, R²)
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2

def preprocess_data(df):
    """
    Tiền xử lý dữ liệu, tạo các đặc trưng thời gian.
    
    Tham số:
        df (pd.DataFrame): DataFrame gốc
        
    Trả về:
        pd.DataFrame: DataFrame sau khi xử lý
    """
    df_processed = df.copy()
    df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['dayofweek'] = df_processed['timestamp'].dt.dayofweek
    df_processed['month'] = df_processed['timestamp'].dt.month
    df_processed['is_daytime'] = df_processed['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
    
    return df_processed

def get_linear_features(df):
    """
    Chọn các đặc trưng cơ bản cho mô hình tuyến tính.
    
    Tham số:
        df (pd.DataFrame): DataFrame đã qua tiền xử lý
        
    Trả về:
        pd.DataFrame: DataFrame chỉ chứa các đặc trưng cần thiết
    """
    return df[['o3op1', 'o3op2', 'no2op1', 'no2op2']]

def engineer_features_advanced(df):
    """
    Tạo các đặc trưng nâng cao cho mô hình XGBoost.
    
    Tham số:
        df (pd.DataFrame): DataFrame đã qua tiền xử lý
        
    Trả về:
        pd.DataFrame: DataFrame với các đặc trưng đã tạo
    """
    df_advanced = df.copy()
    
    # Tạo các đặc trưng tương tác và phi tuyến
    df_advanced['temp_humidity_interaction'] = df_advanced['temp'] * df_advanced['humidity']
    df_advanced['temp_squared'] = df_advanced['temp'] ** 2
    df_advanced['o3_interaction'] = df_advanced['o3op1'] * df_advanced['o3op2']
    df_advanced['no2_interaction'] = df_advanced['no2op1'] * df_advanced['no2op2']
    
    # Tạo các đặc trưng tỷ lệ
    df_advanced['o3_ratio'] = df_advanced['o3op1'] / df_advanced['o3op2'].replace(0, 1)
    df_advanced['no2_ratio'] = df_advanced['no2op1'] / df_advanced['no2op2'].replace(0, 1)
    
    return df_advanced

def get_advanced_features(df):
    """
    Chọn các đặc trưng cho mô hình nâng cao.
    
    Tham số:
        df (pd.DataFrame): DataFrame đã qua kỹ thuật tạo đặc trưng
        
    Trả về:
        pd.DataFrame: DataFrame chứa các đặc trưng nâng cao
    """
    features = ['temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2', 
                'hour', 'dayofweek', 'month', 'is_daytime',
                'temp_humidity_interaction', 'temp_squared', 
                'o3_interaction', 'no2_interaction', 'o3_ratio', 'no2_ratio']
    
    return df[features]