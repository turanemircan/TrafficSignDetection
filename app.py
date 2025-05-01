from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import torch
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("C:/Users/Emircan/Documents/Python/Yol_Isareti_Tanima_Sistemi/runs/detect/train/weights/best.pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        img = Image.open(img_file).convert('RGB')

        results = model.predict(img, device=0 if device.type == 'cuda' else 'cpu')
        result = results[0]

        if len(result.boxes.cls) > 0:
            confs = result.boxes.conf.tolist()
            labels = result.boxes.cls.tolist()
            max_index = confs.index(max(confs))
            confidence = confs[max_index] * 100
            class_id = int(labels[max_index])
            predicted_class = result.names[class_id]

            if confidence < 55:
                predicted_class = "Tespit edilemedi"
        else:
            predicted_class = "Tespit edilemedi"
            confidence = 0

        # Inference süresi
        inference_time = result.speed['inference']

        # Görüntüyü kaydet ve yolu oluştur
        save_path = os.path.join('static/uploads', img_file.filename)
        img.save(save_path)

        return render_template('index.html',
                               image_path=save_path,
                               predicted_class=predicted_class,
                               confidence=confidence,
                               inference_time=inference_time)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
