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

# Etiketler (modelin sınıf sıralamasına göre!)
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

# Softmax fonksiyonu
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Ana kontrol endpoint
@app.route('/', methods=['GET'])
def index():
    return "✅ Activity Detection API is running!", 200

# Sadece sınıf döndüren endpoint
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

# Olasılıkları da döndüren yeni endpoint
@app.route('/probabilities', methods=['POST'])
def predict_with_probabilities():
    try:
        data = request.json
        features = data['features']

        input_data = np.array(features, dtype=np.float32).reshape(1, 562, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

        probs = softmax(raw_output)  # ✅ normalize et
        probabilities = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        predicted_class = LABELS[int(np.argmax(probs))]

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
