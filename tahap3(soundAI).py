import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import threading
from gtts import gTTS
import pygame
import io

# Muat model yang telah dilatih dari folder 'hasil_pelatihan'
model_path = 'hasil_pelatihan/gesture_recognition_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model berhasil dimuat.")
else:
    print(f"Model tidak ditemukan di {model_path}")
    exit()

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi pygame mixer untuk memutar suara
pygame.mixer.init()

# Fungsi untuk menjalankan TTS di thread terpisah dan memutar suara langsung
def speak_text(text):
    tts = gTTS(text=text, lang='id')  # Tentukan bahasa Indonesia
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)  # Pindahkan pointer ke awal file

    pygame.mixer.music.load(audio_data, 'mp3')
    pygame.mixer.music.play()

    # Tunggu hingga suara selesai diputar
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Variabel untuk melacak gerakan terakhir yang dibacakan dan buffer prediksi
last_spoken_gesture = None
gesture_buffer = []
BUFFER_SIZE = 5  # Ukuran buffer untuk memastikan prediksi stabil

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    result = hands.process(rgb_frame)

    # Jika tangan terdeteksi, ambil landmark dan prediksi gerakan
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ekstrak koordinat landmark tangan
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Normalisasi landmark untuk konsistensi
            x_min, y_min = np.min(landmarks[:, :2], axis=0)
            x_max, y_max = np.max(landmarks[:, :2], axis=0)
            landmarks[:, :2] = (landmarks[:, :2] - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])

            # Rata landmark menjadi array 1D
            landmarks = landmarks.flatten()

            # Prediksi gerakan menggunakan model
            prediction = model.predict([landmarks])
            gesture_name = prediction[0]

            # Tambahkan prediksi ke buffer
            gesture_buffer.append(gesture_name)
            if len(gesture_buffer) > BUFFER_SIZE:
                gesture_buffer.pop(0)  # Hapus elemen tertua di buffer

            # Periksa jika prediksi stabil di buffer
            if gesture_buffer.count(gesture_name) == BUFFER_SIZE:
                # Prediksi stabil, lakukan tindakan
                if gesture_name != last_spoken_gesture:
                    threading.Thread(target=speak_text, args=(gesture_name,)).start()
                    last_spoken_gesture = gesture_name  # Update gerakan terakhir yang dibacakan

            # Tampilkan hasil prediksi di layar
            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Gesture Recognition", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
