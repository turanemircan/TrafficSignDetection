from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import torch
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/signs'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(r"C:\Users\Emircan\Documents\Python\Yol_Isareti_Tanima_Sistemi\runs\detect\train\weights\best.pt")

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

        example_sign_path = None
        if predicted_class != "Tespit edilemedi":
            folder_path = app.config['RESULT_FOLDER']
            for f in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, f)):
                    name_without_ext = os.path.splitext(f)[0]
                    if predicted_class == name_without_ext:
                        example_sign_path = f"signs/{f}" 
                        break

        inference_time = result.speed['inference']

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img.save(save_path)

        return render_template('index.html',
                               image_path=save_path,
                               predicted_class=predicted_class,
                               confidence=confidence,
                               inference_time=inference_time,
                               example_sign_path=example_sign_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
