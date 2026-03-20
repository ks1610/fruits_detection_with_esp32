import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Cấu hình tham số tối ưu cho vi điều khiển
IMG_SIZE = 96
BATCH_SIZE = 32
DATA_PATH = 'fruits'

# Chuẩn hóa giá trị pixel về [0, 1]
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    f'{DATA_PATH}/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    f'{DATA_PATH}/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Test gen cho bước đánh giá
test_gen = datagen.flow_from_directory(
    f'{DATA_PATH}/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Tải MobileNetV1 (bỏ lớp Output mặc định)
base_model = MobileNet(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.25
)

# Đóng băng các lớp cơ sở (Transfer Learning)
base_model.trainable = False

# Xây dựng phần phân loại cho 4 loại quả
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x) # 4 lớp: apple, banana, orange, pineapple

model = Model(inputs=base_model.input, outputs=predictions)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Bắt đầu huấn luyện
print("Bắt đầu huấn luyện...")
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=15
)

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