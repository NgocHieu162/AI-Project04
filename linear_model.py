from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import TEST_SIZE, RANDOM_STATE
from utils import load_data, get_linear_features, evaluate_model, save_model


def train_linear_models(X_train, y_train, target_name):
    """
    Huấn luyện và đánh giá các mô hình tuyến tính.
    
    Tham số:
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện
        y_train (pd.Series): Biến mục tiêu huấn luyện
        target_name (str): Tên của biến mục tiêu ('Ozone' hoặc 'NO2')
        
    Trả về:
        tuple: (mô hình tốt nhất, tên mô hình tốt nhất, MAE)
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(max_iter=5000, tol=1e-3,)
    }
    
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    best_mae = float('inf')
    best_model_name = None
    best_model = None
    
    print(f"\n--- {target_name} Prediction Models ---")
    
    for name, model in models.items():
        if name in ['Ridge Regression', 'Lasso Regression']:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"{name} best alpha: {grid_search.best_params_['alpha']}")
        else:
            model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        _, train_mae, _ = evaluate_model(model, X_train, y_train)
        
        print(f"Training MAE: {train_mae:.4f}, Model: {name}")
        
        if train_mae < best_mae:
            best_mae = train_mae
            best_model_name = name
            best_model = model
    
    # Hiển thị hệ số của mô hình tốt nhất
    coef = best_model.coef_
    intercept = best_model.intercept_
    
    print(f"\nBest model for {target_name}: {best_model_name} with MAE: {best_mae:.4f}")
    print("Model Coefficients:")
    feature_names = X_train.columns
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {coef[i]:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    
    return best_model, best_model_name, best_mae


def run_linear_model(data_path):
    """
    Chạy mô hình tuyến tính.
    
    Tham số:
        data_path (str): Đường dẫn đến file dữ liệu
    """
    print("\n=== RUNNING LINEAR MODEL ===")
    
    # Đọc dữ liệu
    df = load_data(data_path)
    
    # Lấy các đặc trưng cơ bản
    X = get_linear_features(df)
    y_ozone = df['OZONE']
    y_no2 = df['NO2']
    
    # Chia dữ liệu
    X_train, X_test, y_ozone_train, y_ozone_test = train_test_split(
        X, y_ozone, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    _, _, y_no2_train, y_no2_test = train_test_split(
        X, y_no2, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True)
    
    # Huấn luyện và đánh giá mô hình Ozone
    best_ozone_model, best_ozone_model_name, best_ozone_mae = train_linear_models(
        X_train, y_ozone_train, "Ozone (O₃)"
    )
    
    # Huấn luyện và đánh giá mô hình NO2
    best_no2_model, best_no2_model_name, best_no2_mae = train_linear_models(
        X_train, y_no2_train, "NO₂"
    )
    
    # Lưu mô hình
    save_model(best_ozone_model, 'models/model_Ozone_Task1.pkl')
    save_model(best_no2_model, 'models/model_NO2_Task1.pkl')
    
    # Tóm tắt
    print("\n----- SUMMARY -----")
    print(f"Best Ozone (O₃) model: {best_ozone_model_name}")
    print(f"Ozone (O₃) Training MAE: {best_ozone_mae:.4f}")
    print(f"\nBest NO₂ model: {best_no2_model_name}")
    print(f"NO₂ Training MAE: {best_no2_mae:.4f}")



