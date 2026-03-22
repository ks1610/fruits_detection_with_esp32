#include <WiFi.h>
#include <WebServer.h>
#include "model_data.h" 

// --- TENSORFLOW LITE INCLUDES ---
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

const char* ssid = "ESP32_Fruit_AI";
const char* password = "";

WebServer server(80);

// --- CẤU HÌNH TENSORFLOW LITE ---
const int kTensorArenaSize = 105 * 1024;
uint8_t* tensor_arena = nullptr;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* CLASSES[] = {"Táo", "Chuối", "Cam", "Dứa"};

// Biến toàn cục đếm số lượng byte ảnh đã truyền thẳng vào AI
int received_bytes = 0; 

// --- GIAO DIỆN WEB ---
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32 Fruit AI</title>
  <style>
    body { text-align: center; font-family: Arial; padding: 20px; }
    video, canvas { max-width: 100%; border: 2px solid #333; border-radius: 8px; }
    button { padding: 15px 30px; font-size: 18px; margin-top: 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
    #result { font-size: 24px; font-weight: bold; margin-top: 20px; color: #28a745; }
  </style>
</head>
<body>
  <h2>ESP32 AI Camera</h2>
  <video id="video" autoplay playsinline></video>
  <canvas id="canvas" width="96" height="96" style="display:none;"></canvas>
  <br>
  <button onclick="captureAndSend()">Nhận Diện</button>
  <div id="result">Đang chờ...</div>

  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resultDiv = document.getElementById('result');

    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
      .then(stream => { video.srcObject = stream; })
      .catch(err => { resultDiv.innerText = "Lỗi Camera: " + err; });

    async function captureAndSend() {
      resultDiv.innerText = "Đang truyền dữ liệu...";
      ctx.drawImage(video, 0, 0, 96, 96);
      
      const imgData = ctx.getImageData(0, 0, 96, 96).data;
      const rgbData = new Uint8Array(96 * 96 * 3);
      let j = 0;
      for (let i = 0; i < imgData.length; i += 4) {
        rgbData[j++] = imgData[i];     // R
        rgbData[j++] = imgData[i+1];   // G
        rgbData[j++] = imgData[i+2];   // B
      }

      // TỐI ƯU JS: Gói ảnh thành một file nhị phân (Blob) để truyền qua luồng Streaming
      const blob = new Blob([rgbData], { type: 'application/octet-stream' });
      const formData = new FormData();
      formData.append('image', blob, 'image.bin');

      try {
        // Trình duyệt sẽ tự động chia nhỏ file ra gửi, không làm ngộp ESP32
        const response = await fetch('/infer', {
          method: 'POST',
          body: formData
        });
        const text = await response.text();
        resultDiv.innerText = "Kết quả: " + text;
      } catch (err) {
        resultDiv.innerText = "Lỗi kết nối!";
      }
    }
  </script>
</body>
</html>
)rawliteral";

// --- HÀM 1: ĐÓN DỮ LIỆU TỪNG MẢNH VÀ NẠP THẲNG VÀO TENSOR (Tránh tạo String) ---
void handleUpload() {
  HTTPUpload& upload = server.upload();
  
  if (upload.status == UPLOAD_FILE_START) {
    received_bytes = 0;
    Serial.println("\nĐang nhận ảnh trực tiếp vào AI...");
    
  } else if (upload.status == UPLOAD_FILE_WRITE) {
    // Chép từng byte ngay khi nó vừa đến card WiFi vào thẳng bộ nhớ mô hình
    for (size_t i = 0; i < upload.currentSize; i++) {
      if (received_bytes < 96 * 96 * 3) {
        input->data.int8[received_bytes] = (int8_t)(((uint8_t)upload.buf[i]) - 128);
        received_bytes++;
      }
    }
  }
}

// --- HÀM 2: XỬ LÝ NHẬN DIỆN SAU KHI ĐÃ NHẬN ĐỦ ẢNH ---
void handleInference() {
  if (received_bytes != 96 * 96 * 3) {
    server.send(400, "text/plain", "Kích thước ảnh bị lỗi trong lúc truyền");
    return;
  }

  Serial.println("Đã nhận đủ ảnh. BẮT ĐẦU TÍNH TOÁN AI...");
  
  long start_time = millis();
  TfLiteStatus status = interpreter->Invoke(); // Bấm còi cho AI chạy
  long end_time = millis();

  Serial.print("Thời gian tính toán: ");
  Serial.print(end_time - start_time);
  Serial.println(" ms");

  if (status != kTfLiteOk) {
    Serial.println("Lỗi AI Invoke!");
    server.send(500, "text/plain", "Lỗi AI nội bộ");
    return;
  }

  int8_t max_val = output->data.int8[0];
  int max_idx = 0;
  for (int i = 1; i < 4; i++) {
    if (output->data.int8[i] > max_val) {
      max_val = output->data.int8[i];
      max_idx = i;
    }
  }

  //check confidence level
  float confidence = (max_val + 128.0) / 256.0 * 100.0;
  Serial.print("Độ chính xác: ");
  Serial.print(confidence);
  Serial.println("%");
  String result_str = "";
  float THRESHOLD = 65.0;
  
  if (confidence < THRESHOLD) {
    result_str = "Vật thể lạ / Không rõ (" + String(confidence, 1) + "%)";
  } else {
    result_str = String(CLASSES[max_idx]) + " (" + String(confidence, 1) + "%)";
  }

  Serial.print("Hoàn thành! Kết quả: ");
  Serial.println(result_str);
  server.send(200, "text/plain", result_str);
}

void setup() {
  Serial.begin(115200);
  
  // Cấp phát dư 16 byte và ép căn lề 16-byte cho AI
  uint8_t* raw_tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize + 16, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (raw_tensor_arena == NULL) {
    Serial.println("Lỗi: Không đủ bộ nhớ Heap!");
    while(1);
  }
  tensor_arena = (uint8_t*)(((uintptr_t)raw_tensor_arena + 15) & ~15);
  
  // Khởi tạo AI
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Lỗi phiên bản mô hình TFLite!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Lỗi cấp phát vùng nhớ Arena!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite khởi tạo thành công!");

  // Phát WiFi
  WiFi.softAP(ssid, password);
  Serial.print("Đã phát Wi-Fi. IP: ");
  Serial.println(WiFi.softAPIP());

  // Định tuyến Web
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/html", index_html);
  });

  // TỐI ƯU CẤU TRÚC: Chèn thêm hàm handleUpload để nhận dữ liệu song song
  server.on("/infer", HTTP_POST, handleInference, handleUpload);
  
  server.begin();
  Serial.println("Web server đang chạy...");
}

void loop() {
  server.handleClient();
}