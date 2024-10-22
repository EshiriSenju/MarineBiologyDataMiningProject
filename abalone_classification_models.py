import pandas as pd  
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
file_path = './cleaned_abalone_data_2.csv'
abalone_data = pd.read_csv(file_path)

# Prepare data
X = abalone_data.drop(columns=['Sex'])
y = abalone_data['Sex']

# Convert categorical target variable 'Sex' to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply SMOTE to handle class imbalance with adjusted n_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust n_neighbors to 1
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning with GridSearchCV

# Decision Tree Classifier Grid Search
dt_params = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
dt_model = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
dt_model.fit(X_train_scaled, y_train)
dt_best = dt_model.best_estimator_

# Random Forest Classifier Grid Search
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20], 'min_samples_split': [2, 5, 10]}
rf_model = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
rf_model.fit(X_train_scaled, y_train)
rf_best = rf_model.best_estimator_

# K-Nearest Neighbors Grid Search
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_model.fit(X_train_scaled, y_train)
knn_best = knn_model.best_estimator_

# Model Evaluation

# Decision Tree Evaluation
dt_predictions = dt_best.predict(X_test_scaled)
print("Best Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))

# Random Forest Evaluation
rf_predictions = rf_best.predict(X_test_scaled)
print("\nBest Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# KNN Evaluation
knn_predictions = knn_best.predict(X_test_scaled)
print("\nBest K-Nearest Neighbors Classifier")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Classification Report:\n", classification_report(y_test, knn_predictions))

# Confusion Matrix for the best-performing model (Random Forest)
conf_matrix = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Best Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
