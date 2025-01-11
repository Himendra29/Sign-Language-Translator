import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Arrays to store data
data = []
labels = []

# Label for the gesture being recorded
gesture_name = "nafisa"  # Ubah sesuai nama gestur yang sedang dikumpulkan

# Buat folder data_gesture jika belum ada
if not os.path.exists('data_gesture'):
    os.makedirs('data_gesture')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Draw hand landmarks if hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Convert landmarks to numpy array and add to data
            data.append(np.array(landmarks).flatten())
            labels.append(gesture_name)
            
            # Display confirmation message
            print(f"Data for '{gesture_name}' gesture saved. Total samples collected: {len(data)}")

    # Show the frame
    cv2.imshow("Frame", frame)

    # Press 'q' to stop collecting data
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert data to a DataFrame and save to CSV in data_gesture folder
df = pd.DataFrame(data)
df['label'] = labels  # Tambahkan kolom label di akhir DataFrame
file_path = f'data_gesture/{gesture_name}_gesture_data.csv'
df.to_csv(file_path, index=False)  # Simpan sebagai CSV tanpa indeks

print(f"Data berhasil disimpan ke file '{file_path}'.")

# Release resources
cap.release()
cv2.destroyAllWindows()
