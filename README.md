# Group43-Final-Project: Plant Disease Detection Web App 🌿🔬

A full-stack deep learning web application to detect plant leaf diseases for Pepper, Tomato, and Potato using the PlantVillage dataset.

This project uses:
- Convolutional Neural Networks (CNN) with Keras / TensorFlow
- Flask for a web interface
- PlantVillage dataset

---

## 🚀 Features

✅ Upload a leaf image and get predicted disease class  
✅ Shows disease description, symptoms, causes, severity  
✅ Provides treatment recommendations (preventive, organic, chemical)  
✅ Trained on the PlantVillage dataset  
✅ Flask web interface with file upload  

---

## 📂 Project Structure

Group43-Final-Project/
│
├── app.py                 # Flask entrypoint
├── routes.py              # App views and routing
├── ml_model.py            # CNN model logic
├── train_model.py         # Model training script
├── evaluate_model.py      # Model evaluation script
├── disease_data.py        # Disease metadata & treatments
├── requirements.txt       # Python dependencies
├── .env                   # Secret environment variables
├── .gitignore
│
├── models/                # Trained .keras models + class names
├── dataset/               # Training dataset (split by class)
│   ├── Pepper__bell___Bacterial_spot/
│   ├── ...
│   └── test/              # Evaluation dataset (10+ per class)
├── uploads/               # Uploaded user images
├── static/                # Frontend static assets
│   ├── css/
│   └── js/
├── templates/             # HTML pages
│   ├── index.html
│   ├── results.html
│   ├── 404.html
│   └── 500.html
├── logs/                  # TensorBoard and training logs

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

  ✅ Test accuracy
  ✅ Detailed classification report

---

## 🌐 Running the Web App
To start the Flask app:

```bash
python app.py
```

Open your browser:

http://127.0.0.1:5000/

📸 Using the Web Interface:
  1️⃣ Go to /
  2️⃣ Upload a plant leaf image (png, jpg, jpeg, gif, bmp, webp)
  3️⃣ View results with:

  Disease class

  Confidence score

  Description, symptoms, causes, severity

  Treatment recommendations

⚠️ *Notes on Dataset Splits*
Your test set must contain all classes that the model was trained on.

Even 5–10 images per class is sufficient.

This ensures the classification report works correctly without errors or warnings.

---

📜 License
MIT License

---

## 🙏 Acknowledgements:
PlantVillage dataset (Kaggle)

TensorFlow / Keras

Flask

---

For questions or issues, please open an issue on this repository or contact me.