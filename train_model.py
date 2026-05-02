import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*50)
print("EyeClass - Training from Image Folders")
print("="*50)

# ----- 1. Set paths -----
train_path = "data/train"
test_path = "data/test"

# Emotion labels (must match your folder names exactly)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ----- 2. Load images from folders -----
print("\n[1/4] Loading images from folders...")

def load_images(base_path):
    images = []
    labels = []
    
    for label_idx, emotion in enumerate(emotion_labels):
        folder_path = os.path.join(base_path, emotion)
        
        if not os.path.exists(folder_path):
            print(f"      ⚠️  Folder not found: {folder_path}")
            continue
            
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(label_idx)
                    count += 1
        print(f"      ✓ {emotion}: {count} images")
    
    return np.array(images), np.array(labels)

print("\n   Training data:")
X_train, y_train = load_images(train_path)
print("\n   Test data:")
X_test, y_test = load_images(test_path)

# Normalize and reshape
X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print(f"\n   Total training images: {X_train.shape[0]}")
print(f"   Total test images: {X_test.shape[0]}")

# ----- 3. Data augmentation -----
print("\n[2/4] Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)
print("      ✓ Augmentation ready")

# ----- 4. Build the CNN -----
print("\n[3/4] Building CNN architecture...")
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----- 5. Train -----
print("\n[4/4] Training model (this will take 20-40 minutes)...")
print("="*50)

checkpoint = ModelCheckpoint('models/emotion_model.h5', 
                            monitor='val_accuracy', 
                            save_best_only=True, 
                            verbose=1)
early_stop = EarlyStopping(monitor='val_loss', 
                          patience=10, 
                          restore_best_weights=True, 
                          verbose=1)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ----- 6. Final evaluation -----
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"✅ Training Complete!")
print(f"   Test Accuracy: {acc*100:.2f}%")
print(f"   Model saved to: models/emotion_model.h5")
print(f"{'='*50}")