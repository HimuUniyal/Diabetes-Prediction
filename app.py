import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


file_path = 'diabetes.csv'
diabetes_df = pd.read_csv(file_path)


X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier(random_state=42)) 
])


param_grid_small = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}


grid_search_small = GridSearchCV(pipeline, param_grid_small, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)


grid_search_small.fit(X_train, y_train)


best_model_small = grid_search_small.best_estimator_


train_y_pred = best_model_small.predict(X_train)
test_y_pred = best_model_small.predict(X_test)


train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)


with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(best_model_small, model_file)


with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data_nparray = np.asarray(data).reshape(1, -1)

  
    prediction = model.predict(input_data_nparray)
    prediction_proba = model.predict_proba(input_data_nparray)

    result = 'This person does not have diabetes.' if prediction == 0 else 'This person has diabetes.'

    return render_template('index.html', prediction_text=result, 
                           prob_no_diabetes=prediction_proba[0][0], 
                           prob_diabetes=prediction_proba[0][1])

if __name__ == "__main__":
    app.run(debug=True)
