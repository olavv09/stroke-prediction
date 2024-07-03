import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

file_path = './healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)

data = data.drop('id', axis=1).drop('avg_glucose_level', axis=1)
data = data.dropna()
X = data.drop('stroke', axis=1)
y = data['stroke']

categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
transformer = ColumnTransformer([('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

def predict(new_data):
    encoded = transformer.transform(new_data)
    predicted = model.predict(encoded)
    return predicted.round()
