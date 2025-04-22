from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from utils import TEST_SIZE, RANDOM_STATE
from utils import load_data, preprocess_data, engineer_features_advanced, get_advanced_features, evaluate_model, save_model


def train_and_tune_xgboost(X_train, y_train, param_grid=None):
    """
    Huấn luyện và tinh chỉnh mô hình XGBoost.
    
    Tham số:
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện
        y_train (pd.Series): Biến mục tiêu huấn luyện
        param_grid (dict, optional): Lưới tham số cho GridSearchCV
        
    Trả về:
        tuple: (tham số tốt nhất, mô hình tốt nhất)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_estimator_

def run_advanced_model(data_path):
    """
    Chạy mô hình nâng cao với XGBoost.
    
    Tham số:
        data_path (str): Đường dẫn đến file dữ liệu
    """
    print("\n=== RUNNING ADVANCED MODEL ===")
    
    # Đọc dữ liệu
    df = load_data(data_path)
    
    # Tiền xử lý dữ liệu
    df = preprocess_data(df)
    
    # Tạo đặc trưng nâng cao
    df = engineer_features_advanced(df)
    
    # Lấy đặc trưng và biến mục tiêu
    X = get_advanced_features(df)
    y_ozone = df['OZONE']
    y_no2 = df['NO2']
    
    # Chia dữ liệu
    X_train, X_test, y_train_ozone, y_test_ozone = train_test_split(
        X, y_ozone, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    _, _, y_train_no2, y_test_no2 = train_test_split(
        X, y_no2, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    
    # Huấn luyện mô hình cho Ozone
    print("\nTraining Ozone model...")
    best_params_ozone, best_model_ozone = train_and_tune_xgboost(X_train, y_train_ozone)
    
    # Huấn luyện mô hình cho NO2
    print("\nTraining NO2 model...")
    best_params_no2, best_model_no2 = train_and_tune_xgboost(X_train, y_train_no2)
    
    # Lưu mô hình
    save_model(best_model_ozone, 'models/model_Ozone_Task2.pkl')
    save_model(best_model_no2, 'models/model_NO2_Task2.pkl')
    
    print(f"Best parameters for Ozone model: {best_params_ozone}")
    print(f"Best parameters for NO2 model: {best_params_no2}")
    
    # Đánh giá mô hình
    rmse_ozone, mae_ozone, r2_ozone = evaluate_model(best_model_ozone, X_test, y_test_ozone)
    rmse_no2, mae_no2, r2_no2 = evaluate_model(best_model_no2, X_test, y_test_no2)
    
    print("\n=== Ozone Model Performance ===")
    print(f"RMSE: {rmse_ozone:.4f}")
    print(f"MAE: {mae_ozone:.4f}")
    print(f"R²: {r2_ozone:.4f}")
    
    print("\n=== NO2 Model Performance ===")
    print(f"RMSE: {rmse_no2:.4f}")
    print(f"MAE: {mae_no2:.4f}")
    print(f"R²: {r2_no2:.4f}")