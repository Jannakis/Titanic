import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import classification_report

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# Datenübersicht
df = pd.concat([train_data,test_data], axis=0)
#print(df.info())
#print(df.describe())

# Datensatz bereinigen
df = df.drop(columns=["Name", "Ticket","PassengerId","Cabin","Embarked"])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()


# Daten trennen
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Daten skalieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Trainings und Testdatensatz erstellen
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

def grid_search_rf(X_train, X_val, y_train, y_val):
    # Random Forrest Classifiert Parameter finetuning durch Gridsearch
    parameters_rf = {'n_estimators':list(range(5,15)),
                "criterion":['gini', 'entropy', 'log_loss']}
    rf_clf = RandomForestClassifier(random_state=42)
    clf1 = GridSearchCV(rf_clf, parameters_rf)
    clf1.fit(X_train, y_train)
    print(clf1.best_estimator_)

    # Schritt 1: Vorhersagen für die Testdaten
    y_test_pred = clf1.best_estimator_.predict(X_val)
    #Schritt 2: Genauigkeit berechnen
    test_accuracy = accuracy_score(y_val, y_test_pred)
    print("Test Accuracy RF:", test_accuracy)
    print("Classification Report:")
    print(classification_report(y_val, y_test_pred))

################### Model 2 ###############################################
def random_search_rf(X_train, X_val, y_train, y_val, n_iter=150, cv=5, random_state=42):

    param_dist = {
    'n_estimators': np.arange(50, 500, 50),  # Anzahl der Bäume
    'max_depth': [None, 10, 20, 30, 40],  # Tiefe der Bäume
    'min_samples_split': [2, 5, 10],  # Mindestanzahl der Proben, um einen Knoten zu teilen
    'min_samples_leaf': [1, 2, 4],  # Mindestanzahl der Proben, die in einem Blatt verbleiben
    'bootstrap': [True, False]  # Ob Bootstrap beim Ziehen von Proben verwendet werden soll
    }
    rf_clf = RandomForestClassifier(random_state=random_state)

    random_search = RandomizedSearchCV(rf_clf, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, verbose=2)

    random_search.fit(X_train, y_train)
    
    # Ausgabe der besten Parameter, des besten Modells und des besten Scores
    print(f"Beste Parameter: {random_search.best_params_}")
    print(f"Bester Score (Trainingsgenauigkeit): {random_search.best_score_}")

    # Schritt 1: Vorhersagen für die Testdaten
    y_test_pred = random_search.best_estimator_.predict(X_val)
    #Schritt 2: Genauigkeit berechnen
    test_accuracy = accuracy_score(y_val, y_test_pred)
    print("Test Accuracy RF:", test_accuracy)
    print("Classification Report:")
    print(classification_report(y_val, y_test_pred))

#random_search_rf(X_train, X_val, y_train, y_val)
#grid_search_rf(X_train, X_val, y_train, y_val)