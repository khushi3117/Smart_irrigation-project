import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import MinMaxScaler 
import joblib 

# Step 1: LOAD AND PREPROCESS DATA
df = pd.read_csv("irrigation_machine.csv")

# Drop unwanted column
df = df.drop('Unnamed: 0', axis=1)

# DEFINE FEATURES AND LABELS
X = df.iloc[:, 0:20]  # sensor_0 to sensor_19
Y = df.iloc[:, 20:]   # parcel_0 to parcel_2

# Scaling features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train classifier
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model = MultiOutputClassifier(rf)
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=Y.columns))

# Parcel ON summary
print(df[['parcel_0', 'parcel_1', 'parcel_2']].sum())

# Parcel activation visualization
conditions = {
    "Parcel 0 ON": df['parcel_0'],
    "Parcel 1 ON": df['parcel_1'],
    "Parcel 2 ON": df['parcel_2'],
    "Parcel 0 & 1 ON": df['parcel_0'] & df['parcel_1'],
    "Parcel 0 & 2 ON": df['parcel_0'] & df['parcel_2'],
    "Parcel 1 & 2 ON": df['parcel_1'] & df['parcel_2'],
    "All Parcels ON": df['parcel_0'] & df['parcel_1'] & df['parcel_2'],
}

fig, axs = plt.subplots(nrows=len(conditions), figsize=(10, 15), sharex=True)
for ax, (title, condition) in zip(axs, conditions.items()):
    ax.step(df.index, condition.astype(int), where='post', linewidth=1)
    ax.set_title(f"Sprinkler - {title}")
    ax.set_ylabel("Status")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])

axs[-1].set_xlabel("Time Index (Row Number)")
plt.tight_layout()
plt.show()

# Pump overlap plot
any_pump_on = (df['parcel_0'] == 1) | (df['parcel_1'] == 1) | (df['parcel_2'] == 1)

plt.figure(figsize=(15, 6))
plt.step(df.index, df['parcel_0'], where='post', linewidth=2, label='Parcel 0 Pump', color='blue')
plt.step(df.index, df['parcel_1'], where='post', linewidth=2, label='Parcel 1 Pump', color='orange')
plt.step(df.index, df['parcel_2'], where='post', linewidth=2, label='Parcel 2 Pump', color='green')

plt.title("Pump Activity and Combined Farm Coverage")
plt.xlabel("Time Index (Row Number)")
plt.ylabel("Status")
plt.yticks([0, 1], ['OFF', 'ON'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, "Farm_Irrigation_System.pkl")

