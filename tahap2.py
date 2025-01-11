import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Gabungkan data dari berbagai sumber (misalnya, beberapa file CSV)
data1 = pd.read_csv('data_gesture/halo_gesture_data.csv')
data2 = pd.read_csv('data_gesture/oke_gesture_data.csv')
data3 = pd.read_csv('data_gesture/i love you_gesture_data.csv')
data4 = pd.read_csv('data_gesture/A_gesture_data.csv')
data5 = pd.read_csv('data_gesture/B_gesture_data.csv')
data6 = pd.read_csv('data_gesture/C_gesture_data.csv')
data7 = pd.read_csv('data_gesture/D_gesture_data.csv')
data8 = pd.read_csv('data_gesture/E_gesture_data.csv')
data9 = pd.read_csv('data_gesture/F_gesture_data.csv')
data10 = pd.read_csv('data_gesture/G_gesture_data.csv')
data11 = pd.read_csv('data_gesture/H_gesture_data.csv')
data12 = pd.read_csv('data_gesture/I_gesture_data.csv')
data13 = pd.read_csv('data_gesture/K_gesture_data.csv')
data14 = pd.read_csv('data_gesture/L_gesture_data.csv')
data15 = pd.read_csv('data_gesture/M_gesture_data.csv')
data16 = pd.read_csv('data_gesture/N_gesture_data.csv')
data17 = pd.read_csv('data_gesture/O_gesture_data.csv')
data18 = pd.read_csv('data_gesture/U_gesture_data.csv')
data19 = pd.read_csv('data_gesture/V_gesture_data.csv')
data20 = pd.read_csv('data_gesture/W_gesture_data.csv')
data21 = pd.read_csv('data_gesture/X_gesture_data.csv')
data22 = pd.read_csv('data_gesture/Y_gesture_data.csv')
data23 = pd.read_csv('data_gesture/T_gesture_data.csv')
data24 = pd.read_csv('data_gesture/Q_gesture_data.csv')




data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, 
data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, 
data21, data22, data23, data24], ignore_index=True)


# Pastikan data sudah benar, periksa kolom label dan fitur
print(data.head())

# Pisahkan fitur dan label
X = data.drop('label', axis=1)  # Fitur adalah semua kolom selain label
y = data['label']  # Label adalah kolom 'label'

# Bagi data menjadi pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Simpan model yang telah dilatih
model_path = 'hasil_pelatihan/gesture_recognition_model.joblib'
joblib.dump(model, model_path)
print(f"Model berhasil disimpan di {model_path}")
