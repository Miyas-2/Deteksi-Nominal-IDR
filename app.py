import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOT_FOLDER'] = 'static/plots'

# Settings for LBP
radius = 2
n_points = 4 * radius
METHOD = 'default'

# Initialize variables
image_list = []
image_hist = []
datasetUang = None
kelasUang = None
akurasi = 0
model_gnb = None  # Add global model variable

def initialize_system():
    global image_list, image_hist, datasetUang, kelasUang, akurasi
    
    # Load and process dataset images
    image_list = []
    image_hist = []
    kelas = ""
    
    for i in range(0, 8):
        if i == 0: kelas = "1RIBU"
        elif i == 1: kelas = "2RIBU"
        elif i == 2: kelas = "5RIBU"
        elif i == 3: kelas = "10RIBU"
        elif i == 4: kelas = "20RIBU"
        elif i == 5: kelas = "50RIBU"
        elif i == 6: kelas = "75RIBU"
        elif i == 7: kelas = "100RIBU"
        else: continue
        
        list_1 = []
        for j in range(0, 15):
            file = f"rupiah-currency-image-recognition-system/DATASET_UANG/{kelas}/{j+1}.jpg"
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            
            # LBP processing
            lbp = local_binary_pattern(img, n_points, radius, METHOD)
            hist, bins = np.histogram(lbp.ravel(), 256, [0, 256])
            histt = np.transpose(hist[0:256, np.newaxis])
            image_hist.append(histt)
            
            # Save resized image for display
            pil_img = Image.open(file)
            pil_img = pil_img.resize((470, 200), Image.Resampling.LANCZOS)
            save_path = f"static/dataset_images/{kelas}_{j+1}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pil_img.save(save_path)
            list_1.append(save_path)
            
        image_list.append(list_1)
    
    # Prepare dataset for training
    datasetUang = np.concatenate((image_hist), axis=0).astype(np.float32)
    kelasUang = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
    ]).astype(np.float32)
    
    # Calculate accuracy
    akurasi = calculate_accuracy()
    
    # Create plot
    create_plot()
def calculate_accuracy():
    global model_gnb  # Add global keyword
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        datasetUang, kelasUang, test_size=0.20, random_state=0, stratify=kelasUang)
    
    # Gaussian Naive Bayes
    model_gnb = GaussianNB()
    model_gnb.fit(X_train, y_train)
    model_gnb.fit(X_train, y_train)
    
    # K-Fold Cross Validation
    skf_nb = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model_gnb, datasetUang, kelasUang, cv=skf_nb, scoring='accuracy')
    
    return scores.mean()

def create_plot():
    plt.figure()
    
    # Plot each class with different colors
    sred = datasetUang[kelasUang.ravel()==1]
    plt.scatter(sred[:,0], sred[:,1], 80, 'r', '*')
    
    sblue = datasetUang[kelasUang.ravel()==2]
    plt.scatter(sblue[:,0], sblue[:,1], 80, 'b', '*')    
    
    smagenta = datasetUang[kelasUang.ravel()==3]
    plt.scatter(smagenta[:,0], smagenta[:,1], 80, 'm', '*')    
    
    sblack = datasetUang[kelasUang.ravel()==4]
    plt.scatter(sblack[:,0], sblack[:,1], 80, '#E91E63', '*')    
    
    spink = datasetUang[kelasUang.ravel()==5]
    plt.scatter(spink[:,0], spink[:,1], 80, 'c', '*')    
    
    sgr = datasetUang[kelasUang.ravel()==6]
    plt.scatter(sgr[:,0], sgr[:,1], 80, 'g', '*')    
    
    syl = datasetUang[kelasUang.ravel()==7]
    plt.scatter(syl[:,0], syl[:,1], 80, 'y', '*')
    
    bl = datasetUang[kelasUang.ravel()==8]
    plt.scatter(bl[:,0], bl[:,1], 80, '#FFA500', '*')
    
    # Save plot
    os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)
    plt.savefig(os.path.join(app.config['PLOT_FOLDER'], 'plotData.jpg'))

def predict_currency(image_path):
    # Pra-pemrosesan gambar
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # 1. Deteksi awal apakah ini mungkin uang
    if not is_likely_currency(img):
        return "Bukan Uang Rupiah", 0.0
    
    # 2. Ekstraksi fitur LBP
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), 256, [0,256])
    testData = np.transpose(hist[0:256, np.newaxis])
    
    # 3. Prediksi dengan confidence
    proba = model_gnb.predict_proba(testData)[0]
    max_proba = max(proba)
    predicted_class = model_gnb.predict(testData)[0]
    
    # 4. Terapkan threshold confidence
    if max_proba < 0.6:  # Hanya terima jika confidence > 60%
        return "Bukan Uang Rupiah", max_proba
    
    # Mapping kelas ke nominal uang
    currency_map = {1:"1.000", 2:"2.000", 3:"5.000", 4:"10.000",
                   5:"20.000", 6:"50.000", 7:"75.000", 8:"100.000"}
    return currency_map.get(predicted_class, "Bukan Uang Rupiah"), max_proba

def is_likely_currency(img):
    """Fungsi untuk deteksi awal karakteristik uang"""
    # 1. Cek ukuran
    h, w = img.shape
    if h < 100 or w < 200:  # Ukuran terlalu kecil untuk uang
        return False
    
    # 2. Cek banyaknya tepian (edge density)
    edges = cv.Canny(img, 100, 200)
    edge_ratio = np.sum(edges > 0) / (h * w)
    if edge_ratio < 0.05 or edge_ratio > 0.5:  # Terlalu halus/terlalu banyak tepi
        return False
    
    return True
@app.route('/')
def home():
    return render_template('home.html', akurasi=akurasi)

@app.route('/identify', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Pastikan direktori upload ada
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Generate nama file unik
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4())[:8] + '_' + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Simpan file
        file.save(filepath)
        
        # Process the image
        result = predict_currency(filepath)
        
        # Resize the uploaded image for display
        img = Image.open(filepath)
        img = img.resize((235, 100), Image.Resampling.LANCZOS)
        display_filename = 'display_' + unique_filename
        display_path = os.path.join(app.config['UPLOAD_FOLDER'], display_filename)
        img.save(display_path)
        
        result, confidence = predict_currency(filepath)
    
        return render_template('result.html',
                         result=result,
                         confidence=f"{confidence*100:.2f}%",
                         image_path=display_filename,
                         plot_path='plotPred.jpg')
@app.route('/dataset')
def dataset():
    # Default to first image of first class
    class_idx = 0
    image_idx = 0
    image_path = image_list[class_idx][image_idx]
    image_name = f"Image {image_idx+1}.jpg"
    
    return render_template('dataset.html', 
                         image_path=image_path,
                         image_name=image_name,
                         class_idx=class_idx,
                         image_idx=image_idx)

@app.route('/dataset/<int:class_idx>/<int:image_idx>')
def dataset_image(class_idx, image_idx):
    if class_idx < 0 or class_idx >= len(image_list):
        return redirect(url_for('dataset'))
    
    if image_idx < 0 or image_idx >= len(image_list[class_idx]):
        return redirect(url_for('dataset'))
    
    image_path = image_list[class_idx][image_idx]
    image_name = f"Image {image_idx+1}.jpg"
    
    return render_template('dataset.html', 
                         image_path=image_path,
                         image_name=image_name,
                         class_idx=class_idx,
                         image_idx=image_idx)

@app.route('/plot')
def plot():
    return render_template('plot.html', akurasi=akurasi)

if __name__ == '__main__':
    # Initialize the system
    initialize_system()
    
    # Run the app
    app.run(debug=True)