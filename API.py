import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="uci_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Etiketler (output sırasına göre)
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

# Render kontrol endpoint'i
@app.route('/', methods=['GET'])
def index():
    return "✅ Activity Detection API is running!", 200

# Tahmin endpoint'i (sadece sınıf)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        print("📥 Gelen veri:", features)
        print("📊 Özellik boyutu:", len(features))

        input_data = np.array(features, dtype=np.float32).reshape(1, 562, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output_data))
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        print("🛑 Tahmin sırasında hata:", e)
        return jsonify({'error': str(e)}), 500

# Tahmin + probability endpoint'i
@app.route('/probabilities', methods=['POST'])
def predict_with_probabilities():
    try:
        data = request.json
        features = data['features']
        input_data = np.array(features, dtype=np.float32).reshape(1, 562, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        probabilities = {LABELS[i]: float(output_data[i]) for i in range(len(LABELS))}
        predicted_class = LABELS[int(np.argmax(output_data))]

        return jsonify({
            'prediction': predicted_class,
            'probabilities': probabilities
        })
    except Exception as e:
        print("🛑 Tahmin + prob sırasında hata:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✅ TFLite modeli başarıyla yüklendi")
    app.run(debug=True, host='0.0.0.0', port=5050)
