# Group43-Final-Project: Plant Disease Detection Web App 🌿🔬

A full-stack deep learning web application to detect plant leaf diseases for Pepper, Tomato, and Potato using the PlantVillage dataset.

This project uses:
- Convolutional Neural Networks (CNN) with Keras / TensorFlow
- Flask for a web interface
- PlantVillage dataset
- Python 3.11.8

---

## 🚀 Features

✅ Upload a leaf image and get predicted disease class  
✅ Shows disease description, symptoms, causes, severity  
✅ Provides treatment recommendations (preventive, organic, chemical)  
✅ Trained on the PlantVillage dataset  
✅ Flask web interface with file upload  

---

## 📂 Project Structure

```
group43-Final-Project/
│
├── app.py # Main Flask app entrypoint
├── routes.py # Flask routes and views
├── ml_model.py # CNN model class and training logic
├── train_model.py # Command-line training script
├── evaluate_model.py # Command-line evaluation script
├── disease_data.py # Disease info and treatment database
├── requirements.txt # Python package requirements
├── .env # Environment variables (SECRET)
├── .gitignore
│
├── models/ # Saved Keras models (.keras) and class_names.json
│
├── dataset/ # PlantVillage dataset (train/test splits)
│ ├── Pepper__bell___Bacterial_spot/
│ ├── Pepper__bell___healthy/
│ ├── Potato___Early_blight/
│ └── ...
| └── test/ # For test/evaluation(At least 10 images each folder)
|       ├── Pepper__bell___Bacterial_spot/
│       ├── Pepper__bell___healthy/
│       ├── Potato___Early_blight/
|       └── ...
│
├── uploads/ # Uploaded images (temporary)
│
├── static/ # Static frontend files
│ ├── css/
│ └── js/
│
├── templates/ # HTML templates
│ ├── index.html
│ ├── results.html
│ ├── 404.html
│ └── 500.html
│
└── logs/ # TensorBoard logs and training logs
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/MichaelWaruiru/Group43-Final-Project.git
cd Group43-Final-Project
```


### 2️⃣ Install Python Dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3️⃣ Create Environment Variables
Create a .env file in the root folder with your Flask session secret:
    SESSION_SECRET=your_super_secret_key_here

---

## 🏋️‍♂️ Training the Model
Make sure you have the PlantVillage dataset in the dataset/ folder, structured like:
    dataset/
      Pepper__bell___Bacterial_spot/
      Pepper__bell___healthy/
      Potato___Early_blight/
      ...

Then train the model:

```bash
python train_model.py
```

This will:
  1. Train the CNN or MobileNetV2 transfer model
  2. Save the best model to models/plant_disease_cnn_model.h5
  3. Save class names to models/class_names.json
  4. Plot and save training history as training_history.png


## 📈 Evaluating the Model

You can evaluate the trained model on a separate test dataset:

Make sure your test set also has the same classes:
  dataset/test/
    Pepper__bell___Bacterial_spot/
    Pepper__bell___healthy/
    ...

*NOTE: The test set must also have the same classes*
*Tip: Just ensure in test there are few images to ensure each class is represented*

Then run:
```bash
python evaluate_model.py
```

You will see:

  - Test accuracy
  - Detailed classification report

---

## 🌐 Running the Web App
To start the Flask app:

```bash
python app.py
```

Open your browser:

http://127.0.0.1:5000/

📸 Using the Web Interface:
  1. Go to /
  2. Upload a plant leaf image (png, jpg, jpeg, gif, bmp, webp)
  3.  View results with:

  - Disease class

  - Confidence score

  - Description, symptoms, causes, severity

  Treatment recommendations

⚠️ *Notes on Dataset Splits*
Your test set must contain all classes that the model was trained on.

Even 5–10 images per class is sufficient.

This ensures the classification report works correctly without errors or warnings.

---

📜 License
MIT License

---

## 🙏 Acknowledgements
PlantVillage dataset (Kaggle)

TensorFlow / Keras

Flask

---

For questions or issues, please open an issue on this repository or contact me.