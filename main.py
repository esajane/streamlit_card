# Import Statements
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data.dropna()

def scale_features(data):
    scaler = StandardScaler()
    features = ['Time', 'Amount'] + ['V' + str(i) for i in range(1, 29)]
    X = data[features]
    y = data['Class']
    X_scaled = scaler.fit_transform(X)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return X_scaled, y, features

def plot_distribution(y, title):
    classes, counts = np.unique(y, return_counts=True)
    plt.bar(classes, counts, tick_label=['Non-Fraud', 'Fraud'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.show()

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

def balance_data(X_train, y_train):
    sm = SMOTE(sampling_strategy=0.50, k_neighbors=3, random_state=100)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

def build_compile_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=268):
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def predict_model(model, X_test):
    return model.predict(X_test).flatten()

def evaluate_model(y_test, y_pred_prob):
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    return roc_auc, fpr, tpr, pr_auc, precision, recall, thresholds

def optimal_threshold():
    f1_scores = 2 * recall * precision / (recall + precision)
    optimal_idx = np.nanargmax(f1_scores)  
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def plot_curves(fpr, tpr, roc_auc, recall, precision, pr_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# Main Execution Block
if __name__ == "__main__":
    data = load_data('creditcard 2.csv')
    X_scaled, y, features = scale_features(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    X_train_smote, y_train_smote = balance_data(X_train, y_train)
    model = build_compile_model(X_train_smote.shape[1])
    history = train_model(model, X_train_smote, y_train_smote)
    model.save('model.keras')
    y_pred_prob = predict_model(model, X_test)
    roc_auc, fpr, tpr, pr_auc, precision, recall, thresholds = evaluate_model(y_test, y_pred_prob)
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f'PR AUC: {pr_auc:.2f}')
    print(classification_report(y_test, y_pred_prob > optimal_threshold()))
    print(confusion_matrix(y_test, y_pred_prob > optimal_threshold()))

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_smote, y_train_smote)
    with open('random_forest.pkl', 'wb') as model_file:
        pickle.dump(rf, model_file)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    print(confusion_matrix(y_test, y_pred_rf))

    # XGBoost Classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_smote, y_train_smote)
    with open('xgboost.pkl', 'wb') as model_file:
        pickle.dump(rf, model_file)
    y_pred_xgb = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred_xgb))
    print(confusion_matrix(y_test, y_pred_xgb))

    feature_importance = xgb_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

