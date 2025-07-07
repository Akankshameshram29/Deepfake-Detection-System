import os
import glob
import numpy as np
import cv2
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------- CONFIG --------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
contrib_dir = "user_contributions"
model_path = "my_deepfake_detector.h5"
clear_data_after = True  # ⬅ Set to False if you want to keep user files after retraining

# -------- LOAD USER DATA --------
def load_images_from_folder(folder, label):
    data = []
    for img_path in glob.glob(os.path.join(folder, '*')):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            data.append((img, label))
    return data

real_data = load_images_from_folder(os.path.join(contrib_dir, "real"), 0)
fake_data = load_images_from_folder(os.path.join(contrib_dir, "fake"), 1)

data = real_data + fake_data
if len(data) == 0:
    print("❌ No new data found for retraining.")
    exit()

print(f"✅ Found {len(real_data)} real and {len(fake_data)} fake images")

X, y = zip(*data)
X = np.array(X)
y = np.array(y)

# -------- TRAIN-TEST SPLIT --------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# -------- LOAD OR BUILD MODEL --------
if os.path.exists(model_path):
    print("🔁 Loading existing model...")
    model = load_model(model_path)
else:
    print("🛠 Building new model with MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# -------- TRAINING --------
print("🚀 Retraining model...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
save_model(model, model_path)
print(f"✅ Retrained model saved as: {model_path}")

# -------- OPTIONAL: CLEAR USED DATA --------
if clear_data_after:
    for sub in ['real', 'fake']:
        folder = os.path.join(contrib_dir, sub)
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    print("🧹 Cleared used user data after retraining.")
