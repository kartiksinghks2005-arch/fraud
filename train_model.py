import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

print("Preparing data...")
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Model Accuracy:", acc)

print("Saving model...")
joblib.dump(model, "model.pkl")

print("Training Completed 🎉")