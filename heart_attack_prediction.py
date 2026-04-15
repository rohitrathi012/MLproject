import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(
    r"C:\Users\rohit\Downloads\heart_attack_prediction_india.csv")

print(df.head())
print(df.info())


plt.hist(df['Age'], bins=20)
plt.title("Age Distribution of Patients")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Patients")
plt.show()

plt.hist(df['Cholesterol_Level'], bins=20)
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol Level")
plt.ylabel("Number of Patients")
plt.show()

df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.show()

df['Heart_Attack_Risk'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['No Risk', 'High Risk']
)
plt.title("Heart Attack Risk Percentage")
plt.ylabel("")
plt.show()

plt.hexbin(df['Age'], df['Cholesterol_Level'], gridsize=25)
plt.colorbar(label='Number of Patients')
plt.title("Age vs Cholesterol Level")
plt.xlabel("Age")
plt.ylabel("Cholesterol Level")
plt.show()

sns.countplot(x='Gender', hue='Heart_Attack_Risk', data=df)
plt.title("Gender vs Heart Attack Risk")
plt.ylabel("Number of Patients")
plt.legend(title="Heart Attack Risk (0 = No, 1 = Yes)")
plt.show()


df.ffill(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


X = df.drop(['Heart_Attack_Risk', 'Cholesterol_Level'], axis=1)
y_class = df['Heart_Attack_Risk']
y_reg = df['Cholesterol_Level']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.title("Linear Regression: Actual vs Predicted Cholesterol Level")
plt.xlabel("Actual Cholesterol Level")
plt.ylabel("Predicted Cholesterol Level")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)
print("Logistic Regression Accuracy:", log_acc)
print(confusion_matrix(y_test, log_pred))


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_acc)
print(confusion_matrix(y_test, knn_pred))


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_acc)
print(confusion_matrix(y_test, dt_pred))


nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_acc)
print(confusion_matrix(y_test, nb_pred))


models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes']
accuracies = [log_acc, knn_acc, dt_acc, nb_acc]

plt.bar(models, accuracies)
plt.title("Accuracy Comparison of Classification Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc*100:.2f}%", ha='center')

plt.show()
