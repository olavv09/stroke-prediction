import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Wczytanie i przekształcenia danych
file_path = './healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)
data = data.drop('id', axis=1).drop('avg_glucose_level', axis=1)

gender_mapping = {'Other': 2, 'Male': 0, 'Female': 1}
data['gender'] = data['gender'].map(gender_mapping)

married_mapping = {'No': 0, 'Yes': 1}
data['ever_married'] = data['ever_married'].map(married_mapping)

work_mapping = {'Never_worked': 0, 'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'children': 4}
data['work_type'] = data['work_type'].map(work_mapping)

residence_mapping = {'Rural': 0, 'Urban': 1}
data['Residence_type'] = data['Residence_type'].map(residence_mapping)

smoking_mapping = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
data['smoking_status'] = data['smoking_status'].map(smoking_mapping)

data = data.dropna()
data = data[data["gender"] != 2]

X = data.drop('stroke', axis=1)
y = data['stroke']


# Podzial na dane testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definicja modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mse')
# Trenowanie modelu
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=2)

# Export danych
test = data.drop('stroke', axis=1)

data['training_score'] = model.predict(test)

data['training_score'] = (data['training_score']-data['training_score'].min())/(data['training_score'].max()-data['training_score'].min())
data['training_score'] = round(data['training_score'])
data['training_score'] = data['training_score'].astype(int)


gender_mapping = {0: 'Male', 1: 'Female'}
data['gender'] = data['gender'].map(gender_mapping)

data[data["age"] < 1]["age"] = 0

hypertension_mapping = {0: 'No', 1: 'Yes'}
data['hypertension'] = data['hypertension'].map(hypertension_mapping)

heart_mapping = {0: 'No', 1: 'Yes'}
data['heart_disease'] = data['heart_disease'].map(heart_mapping)

married_mapping = {0: 'No', 1: 'Yes'}
data['ever_married'] = data['ever_married'].map(married_mapping)

work_mapping = {0: 'Never_worked', 1: 'Private', 2: 'Self-employed', 3: 'Govt_job', 4: 'children'}
data['work_type'] = data['work_type'].map(work_mapping)

residence_mapping = {0: 'Rural', 1: 'Urban'}
data['Residence_type'] = data['Residence_type'].map(residence_mapping)

smoking_mapping = {0: 'Unknown', 1: 'formerly smoked', 2: 'never smoked', 3: 'smokes'}
data['smoking_status'] = data['smoking_status'].map(smoking_mapping)

stroke_mapping = {0: 'No', 1: 'Yes'}
data['stroke'] = data['stroke'].map(stroke_mapping)

data.to_csv('./result.csv')

def predict(new_data):
    return model.predict(new_data)[0][0]