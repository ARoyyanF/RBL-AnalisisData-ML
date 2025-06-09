import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import random 

# --- 1. Persiapan Data ---
df_raw = pd.read_csv('one_revolution_paths.csv')
planets = ['1 MERCURY BARYCENTER', '2 VENUS BARYCENTER', '3 EARTH BARYCENTER', '4 MARS BARYCENTER', '5 JUPITER BARYCENTER', '6 SATURN BARYCENTER', '7 URANUS BARYCENTER', '8 NEPTUNE BARYCENTER', '9 PLUTO BARYCENTER']
moons = ['301 MOON', '601 MIMAS', '603 TETHYS', '604 DIONE', '605 RHEA', '607 HYPERION', '608 IAPETUS', '609 PHOEBE', '634 POLYDEUCES', '901 CHARON', '902 NIX', '903 HYDRA', '904 KERBEROS', '905 STYX']
df = df_raw[df_raw['name'].isin(planets + moons)].reset_index(drop=True)

# --- 2. REKAYASA FITUR (FEATURE ENGINEERING) ---
print("--- Menambahkan fitur baru: distance_r ---")
df['distance_r'] = np.sqrt(df['x_au']**2 + df['y_au']**2 + df['z_au']**2)

# Gunakan set fitur yang baru dan lebih kaya
X = df[['x_au', 'y_au', 'z_au', 'distance_r']]
print("Data X sekarang memiliki fitur:", X.columns.tolist())

# Buat kolom target
df['planets_only'] = np.where(df['name'].isin(planets), df['name'], 'other')
df['moons_only'] = np.where(df['name'].isin(moons), df['name'], 'other')


# --- 3. Melatih Model Planet-Detector (dengan Fitur Baru) ---
print("\n--- Melatih Model Planet dengan Fitur Baru---")
y_planet = df['planets_only']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_planet, test_size=0.1, random_state=random.randint(0,1000), stratify=y_planet)
smote_p = SMOTE(random_state=69)
X_train_p_smote, y_train_p_smote = smote_p.fit_resample(X_train_p, y_train_p)

# Gunakan RF dengan parameter standar yang sudah terbukti baik
rf_planet = RandomForestClassifier(n_estimators=100, random_state=random.randint(0,1000), n_jobs=-1)
rf_planet.fit(X_train_p_smote, y_train_p_smote)
print("Model Planet selesai dilatih.")


# --- 4. Melatih Model Moon-Detector (dengan Fitur Baru) ---
print("\n--- Melatih Model Bulan dengan Fitur Baru ---")
y_moon = df['moons_only']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_moon, test_size=0.1, random_state=random.randint(0,1000), stratify=y_moon)
smote_m = SMOTE(random_state=42)
X_train_m_smote, y_train_m_smote = smote_m.fit_resample(X_train_m, y_train_m)

rf_moon = RandomForestClassifier(n_estimators=100, random_state=random.randint(0,1000), n_jobs=-1)
rf_moon.fit(X_train_m_smote, y_train_m_smote)
print("Model Bulan selesai dilatih.")


# --- 5. Prediksi Gabungan & Evaluasi ---
print("\n--- Melakukan Prediksi Gabungan ---")
df['pred_feat_eng'] = rf_planet.predict(X)
other_indices = df[df['pred_feat_eng'] == 'other'].index
df.loc[other_indices, 'pred_feat_eng'] = rf_moon.predict(df.loc[other_indices, X.columns])

accuracy_fe = accuracy_score(df['name'], df['pred_feat_eng'])
print(f"\n\nAccuracy Score setelah Feature Engineering: {accuracy_fe:.4f}")

print("\nLaporan Klasifikasi:")
print(classification_report(df['name'], df['pred_feat_eng'], digits=4))

class_names = sorted(moons+planets)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(df['name'], df['pred_feat_eng'], labels=class_names)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - RF + SMOTE + Feature Engineering')
plt.show()
