# 🧠 A PYTORCH-BASED IMAGE CLASSIFIER

This project implements a complete pipeline for training, evaluating, and using a Convolutional Neural Network (CNN) to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It includes model training, visualizations, evaluation metrics, and custom image prediction — all built using **PyTorch**.

---

## 🔍 Features

* ✅ CNN model with batch normalization and dropout
* 📊 Real-time training loss and accuracy tracking
* 📈 Confusion matrix and classification report
* 🖼️ Visualization: class distribution, sample images, t-SNE plots
* 🔬 RGB pixel correlation heatmap
* 💾 Save and load model from disk (`.pth`)
* 📷 Predict custom input images

---

## 📁 Dataset: CIFAR-10

The CIFAR-10 dataset includes **60,000 images** (32x32 RGB) divided into **10 classes**:

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

* 50,000 training images
* 10,000 test images

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/JV-Vigneesh/PyTorch-based-Image-Classifier.git
cd pytorch-image-classifier
```

### 2️⃣ Install Dependencies

Use pip to install required Python packages:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas pillow
```

### 3️⃣ Run the Main Script

```bash
python main.py
```

* If a model checkpoint `cifar10_cnn.pth` exists, it will load the model and skip training.
* Otherwise, it will train the model, evaluate it, and save it.

---

## 🧠 Model Architecture

```
Input: 3 x 32 x 32
→ Conv2D (3→32) + BN + ReLU + MaxPool
→ Conv2D (32→64) + BN + ReLU + MaxPool
→ Conv2D (64→128) + BN + ReLU + MaxPool
→ Global Average Pooling
→ FC (128→512) + ReLU + Dropout
→ FC (512→10)
```

---

## 🧪 Evaluation Output

* 🔹 **Test Accuracy & Loss**
* 🔹 **Normalized Confusion Matrix**
* 🔹 **Classification Report (Precision, Recall, F1-Score)**

---

## 🖼️ Custom Image Prediction

You can place your own `.jpg` or `.png` images in the root folder. Make sure to update this line in the script:

```python
image_paths = ["t.jpg", "p.jpg", "h.jpg", "f.jpg"]
```

These images will be:

* Resized to 32x32
* Normalized using test transforms
* Passed through the trained model
* Displayed along with predicted class

---

## 📊 Visualizations

* 📌 **Bar chart & pie chart** for class distribution
* 🖼️ **16 sample images** with labels
* 🔍 **t-SNE plot** of image feature vectors
* 🔬 **RGB pixel correlation heatmap**
* 🎯 **Training loss & accuracy curves**

---

## 📦 File Structure

```
├── main.ipynb               # Main script with all logic
├── cifar10_cnn.pth       # Saved model (generated after training)
├── t.jpg, p.jpg, ...     # Custom test images (optional)
└── README.md             # Project documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
