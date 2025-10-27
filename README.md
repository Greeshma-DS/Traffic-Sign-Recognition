
# 🚦 Traffic Sign Recognition

![Traffic Sign Example](https://raw.githubusercontent.com/your-username/Traffic-Sign-Recognition/main/sample_image.png)

Traffic Sign Recognition is a **deep learning project** that classifies **43 types of traffic signs** using **Convolutional Neural Networks (CNNs)**. This project includes:

- Data preprocessing pipeline
- CNN model for image classification
- Model evaluation on a test set
- Deployment via **Streamlit** for interactive predictions

It is useful for **autonomous driving systems, road safety applications, and computer vision learning**.

---

## 📝 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Repository Structure](#repository-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Model Architecture](#model-architecture)  
7. [Training & Evaluation](#training--evaluation)  
8. [Streamlit Web App](#streamlit-web-app)  
9. [Results & Screenshots](#results--screenshots)  
10. [Future Work](#future-work)  
11. [Contributing](#contributing)  
12. [License](#license)  

---

## 🔍 Project Overview

This project aims to **automatically recognize traffic signs** from images. The system uses a CNN model trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.  

**Key Features:**

- High accuracy (~95% on test set)
- Real-time predictions through Streamlit interface
- Easy-to-use Jupyter Notebook for training and evaluation
- Preprocessing pipeline that handles large datasets efficiently
- Save/load model for reuse without retraining

---

## 📂 Dataset

- **Source:** [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  
- **Classes:** 43 traffic sign categories (0–42)  
- **Images:** ~50,000 images  
- **Image Size:** 30x30 pixels after preprocessing  
- **Format:** RGB  

**Files in repository:**

- `Train/` – Training images organized by class  
- `Test/` – Test images  
- `Train.csv` – Optional training labels  
- `Test.csv` – Test labels  
- `Meta/` – Additional metadata (optional)

---

## 🗂 Repository Structure

```

Traffic-Sign-Recognition/
│
├── Train/                     # Training images (43 folders: 0–42)
├── Test/                      # Test images
├── Meta/                      # Metadata (optional)
├── Train.csv                  # CSV with training labels
├── Test.csv                   # CSV with test labels
├── traffic_sign_classification.ipynb  # Full training notebook
├── traffic_sign_model.h5      # Saved Keras model
├── app.py                     # Streamlit web app
├── README.md                  # This file

````

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
````

2. Install required packages:

```bash
pip install tensorflow keras streamlit pillow matplotlib pandas scikit-learn
```

3. Optional: Launch Jupyter Notebook to train the model:

```bash
jupyter notebook
```

---

## 🚀 Usage

### 1. Training the Model

* Open `traffic_sign_classification.ipynb`
* Run all cells sequentially:

  * Load and preprocess data
  * Split into training/validation
  * Build CNN model
  * Train model
  * Save trained model as `traffic_sign_model.h5`

### 2. Running the Streamlit App

```bash
streamlit run app.py
```

* Upload an image of a traffic sign
* Get the predicted class and confidence
* Optional: view probabilities for all classes

---

## 🏗 Model Architecture

**Convolutional Neural Network (CNN):**

| Layer     | Filters/Neurons | Kernel Size | Activation | Notes           |
| --------- | --------------- | ----------- | ---------- | --------------- |
| Conv2D    | 32              | 5x5         | ReLU       | Input layer     |
| Conv2D    | 32              | 5x5         | ReLU       |                 |
| MaxPool2D | -               | 2x2         | -          | Pooling layer   |
| Dropout   | -               | -           | -          | Rate 0.25       |
| Conv2D    | 64              | 3x3         | ReLU       |                 |
| Conv2D    | 64              | 3x3         | ReLU       |                 |
| MaxPool2D | -               | 2x2         | -          | Pooling layer   |
| Dropout   | -               | -           | -          | Rate 0.25       |
| Flatten   | -               | -           | -          |                 |
| Dense     | 256             | -           | ReLU       | Fully connected |
| Dropout   | -               | -           | -          | Rate 0.5        |
| Dense     | 43              | -           | Softmax    | Output layer    |

**Optimizer:** Adam
**Loss Function:** Categorical Crossentropy

---

## 📊 Training & Evaluation

* **Training Accuracy:** ~95%
* **Validation Accuracy:** ~95%
* **Test Accuracy:** ~95%

**Sample Accuracy/Loss Curves:**

![Accuracy](https://raw.githubusercontent.com/your-username/Traffic-Sign-Recognition/main/accuracy_plot.png)
![Loss](https://raw.githubusercontent.com/your-username/Traffic-Sign-Recognition/main/loss_plot.png)

---

## 🌐 Streamlit Web App

* Upload any traffic sign image
* Real-time prediction displayed
* Confidence score shown
* Optionally see probabilities for top classes

---

## 🔮 Future Work

* **Data Augmentation:** Increase robustness for rare classes
* **Transfer Learning:** Use pre-trained CNNs (ResNet, MobileNet)
* **Real-time Video Detection:** Detect traffic signs in videos/streams
* **Deployment:** Dockerized app or web-hosted version

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

---

## ⚖️ License

This project is **open-source** and free for **educational and research purposes**.

---

## 📚 References

* [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* [Keras Documentation](https://keras.io/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Deep Learning for Computer Vision](https://www.deeplearningbook.org/)

---

```

---

```
