# 🌫️ Air Quality Sensor Calibration Project

This project calibrates electrochemical sensors to predict ozone (O3) and nitrogen dioxide (NO2) levels using machine learning techniques.

---

## 📁 Project Structure

```
project/
├── data/
│   ├── train.csv              # Training data
│   └── new_data.csv           # (Optional) new data for prediction
├── models/
│   ├── model_Ozone_Task1.pkl  # Linear model (Task 1)
│   ├── model_NO2_Task1.pkl
│   ├── model_Ozone_Task2.pkl  # Advanced model (Task 2)
│   └── model_NO2_Task2.pkl
├── advanced_model.py          # Task 2: Training script
├── predict.py                 # Prediction script
├── utils.py                   # Utility functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Train the Model (Task 2)

To train the advanced model with non-linear features:

```bash
python advanced_model.py
```

This will:

- Train Random Forest models for O3 and NO2
- Save them as `models/advanced_o3_model.pkl` and `models/advanced_no2_model.pkl`

---

## 📊 Make Predictions

Use `predict.py` to predict O3 and NO2 from new sensor data.

### Required format for input CSV:

Must include at least:

```
Time, temp, humidity, o3op1, o3op2, no2op1, no2op2
```

### Examples:

**Predict using the advanced model (Task 2):**

```bash
python predict.py --data data/new_data.csv --model advanced --output predictions.csv
```

**Predict using the linear model (Task 1):**

```bash
python predict.py --data data/new_data.csv --model linear --output linear_predictions.csv
```

---

## 📌 Notes

- You must **train the models first** before running predictions.
- If using the linear model, make sure you have already trained and saved `model_Ozone_Task1.pkl` and `model_NO2_Task1.pkl`.
- The `predict.py` script will load the appropriate model based on the `--model` argument.
- The `--output` argument specifies the name of the output CSV file for predictions.
- The `--data` argument specifies the path to the input CSV file containing sensor data.

---

## 📃 License

This project is for educational use as part of the Introduction to AI course (CSC14003).
