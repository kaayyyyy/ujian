import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)

# Gunakan path absolut agar kompatibel di semua server
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cek dan buat folder upload jika belum ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect('/')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return redirect(url_for('edit_image', filename=filename))

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)

    action = request.args.get('action')

    # Proses filter dari query string (?action=...)
    if action:
        if action == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif action == 'blur':
            image = cv2.GaussianBlur(image, (5, 5), 0)
        elif action == 'canny':
            image = cv2.Canny(image, 100, 200)
        elif action == 'sobel':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            image = cv2.magnitude(sobelx, sobely)
        elif action == 'laplacian':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.Laplacian(gray, cv2.CV_64F)

    # Proses input POST (rotasi, resize, crop)
    if request.method == 'POST':
        if 'rotate' in request.form and request.form['rotate']:
            try:
                angle = float(request.form['rotate'])
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
            except ValueError:
                pass  # Abaikan input yang tidak valid

        if 'resize_width' in request.form and 'resize_height' in request.form:
            try:
                new_width = int(request.form['resize_width'])
                new_height = int(request.form['resize_height'])
                image = cv2.resize(image, (new_width, new_height))
            except ValueError:
                pass

        if 'crop_x' in request.form and 'crop_y' in request.form and 'crop_w' in request.form and 'crop_h' in request.form:
            try:
                x = int(request.form['crop_x'])
                y = int(request.form['crop_y'])
                w = int(request.form['crop_w'])
                h = int(request.form['crop_h'])
                image = image[y:y+h, x:x+w]
            except ValueError:
                pass

    # Simpan hasil edit jika ada perubahan (GET + action atau POST)
    if action or request.method == 'POST':
        edited_filename = f"edited_{filename}"
        edited_path = os.path.join(app.config['UPLOAD_FOLDER'], edited_filename)
        if image.ndim == 2:
            cv2.imwrite(edited_path, image)
        else:
            cv2.imwrite(edited_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        filename = edited_filename

    return render_template('edit.html', filename=filename)


@app.route('/histogram/<filename>')
def show_histogram(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)

    # Pisahkan channel
    channels = {'blue': 0, 'green': 1, 'red': 2}
    hist_filenames = {}

    for color_name, channel_index in channels.items():
        plt.figure()
        histr = cv2.calcHist([image], [channel_index], None, [256], [0, 256])
        plt.plot(histr, color=color_name)
        plt.title(f"Histogram {color_name.capitalize()}")
        plt.xlim([0, 256])

        hist_filename = f"hist_{color_name}_{filename}"
        hist_path = os.path.join(app.config['UPLOAD_FOLDER'], hist_filename)
        plt.savefig(hist_path)
        plt.close()

        hist_filenames[color_name] = hist_filename

    return render_template('histogram.html', hist_filenames=hist_filenames, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)


