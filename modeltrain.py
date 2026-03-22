import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np

# Cấu hình tham số tối ưu cho vi điều khiển
IMG_SIZE = 96
BATCH_SIZE = 32
DATA_PATH = 'fruits'

# 1. TỐI ƯU: TÁCH RIÊNG BỘ SINH DỮ LIỆU VÀ THÊM DATA AUGMENTATION CHO TẬP TRAIN
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Xoay ảnh ngẫu nhiên tối đa 30 độ
    width_shift_range=0.2,  # Dịch chuyển ảnh ngang
    height_shift_range=0.2, # Dịch chuyển ảnh dọc
    zoom_range=0.2,         # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,   # Lật ngang ảnh
    fill_mode='nearest'     # Điền bù các pixel bị khuyết khi xoay/dịch
)

# Tập Val và Test tuyệt đối KHÔNG được Augmentation, chỉ Rescale để đánh giá công bằng
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    f'{DATA_PATH}/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    f'{DATA_PATH}/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_gen.num_classes
print(f"Số lượng nhãn nhận diện: {NUM_CLASSES} - Tên nhãn: {train_gen.class_indices}")

# Tải MobileNetV1
base_model = MobileNet(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.25
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(NUM_CLASSES, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Bắt đầu huấn luyện...")
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=50 
)

# Lấy dữ liệu từ biến history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

# Tạo khung hình vẽ (kích thước 12x5 inch)
plt.figure(figsize=(12, 5))

# 1. Biểu đồ Accuracy (Độ chính xác)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Độ chính xác (Train)', linewidth=2)
plt.plot(epochs_range, val_acc, 'r--', label='Độ chính xác (Validation)', linewidth=2)
plt.title('Độ chính xác qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# 2. Biểu đồ Loss (Hàm mất mát)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Mất mát (Train)', linewidth=2)
plt.plot(epochs_range, val_loss, 'r--', label='Mất mát (Validation)', linewidth=2)
plt.title('Hàm mất mát qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# Căn chỉnh và lưu/hiển thị biểu đồ
plt.tight_layout()
plt.savefig('training_history.png') # Lưu ảnh ra file
# plt.show() # Hiển thị ảnh trên màn hình
# ---------------------------------------

# Tạo hàm sinh dữ liệu đại diện để TFLite biết phạm vi phân bố dữ liệu mà lượng tử hóa
def representative_data_gen():
    for i in range(10): # Lấy 10 batch từ tập train
        images, _ = next(train_gen)
        yield [images.astype(np.float32)]

# Thiết lập Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Ép chặt đầu vào/đầu ra thành định dạng INT8 (Bắt buộc cho phần cứng vi điều khiển)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Chuyển đổi
tflite_model_quant = converter.convert()

# Lưu mô hình TFLite
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quant)
    
print("Đã lưu mô hình dạng tflite!")

# --- PHẦN ĐƯỢC CHỈNH SỬA ĐỂ TỐI ƯU CHO ESP32 ---
def convert_to_c_array(bytes_data, file_name):
    c_array = "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n"
    
    # Ép kiểu căn lề 16-byte (Bắt buộc cho ESP32 để không bị crash khi cấp phát Tensor Arena)
    c_array += "alignas(16) const unsigned char model_data[] = {\n"
    
    # Chuyển đổi bytes sang format hex "0xXX"
    hex_array = [f"0x{b:02x}" for b in bytes_data]
    
    # Ghi 12 bytes trên mỗi dòng cho dễ nhìn
    for i in range(0, len(hex_array), 12):
        c_array += "    " + ", ".join(hex_array[i:i+12]) + ",\n"
            
    c_array += "};\n\n"
    c_array += f"const unsigned int model_data_len = {len(bytes_data)};\n\n"
    c_array += "#endif // MODEL_DATA_H\n"
    
    with open(file_name, "w") as f:
        f.write(c_array)

convert_to_c_array(tflite_model_quant, 'model_data.h')
print("Đã tạo file model_data.h thành công!")