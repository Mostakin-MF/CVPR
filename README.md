# Computer Vision and Pattern Recognition (CVPR)

**Course Work Practice Repository**

**Student Name:** Md. Mostakin Ali  
**Student ID:** 22-48005-2

---

## Mid Term

### Assignments

#### **Assignment 1: KNN (K-Nearest Neighbors) Implementation** 
[View Notebook](Mid/Assignment_1_KNN.ipynb)

**Task:** Implement K-Nearest Neighbors classifier from scratch for animal image classification.

**What Was Done:**
- Loaded and explored animal dataset with multiple classes
- Preprocessed images: resized to 32×32 pixels and converted to grayscale
- Implemented KNN classifier with custom distance metrics
- Performed 5-fold cross-validation with different K values (3-25)
- Compared L1 (Manhattan) and L2 (Euclidean) distance metrics

**Methodology:**
- Image preprocessing using OpenCV (resizing, color conversion)
- Fold-based cross-validation for model evaluation
- Distance-based classification using nearest neighbor voting
- Parameter tuning to find optimal K value

**Tools & Libraries Used:**
- Python, NumPy, OpenCV (cv2), Matplotlib
- Scikit-learn utilities
- Custom KNN implementation (no sklearn classifier)

---

#### **Assignment 2: Neural Network Implementation**
[View Notebook](Mid/Assignment_2_NN.ipynb)

**Task:** Implement a 3-hidden-layer Neural Network from scratch for multi-class classification.

**What Was Done:**
- Generated synthetic 5-class dataset with 2D features (1000 samples total)
- Implemented 3-hidden-layer Neural Network from scratch
- Network architecture: Input(2) → Hidden1(64) → Hidden2(32) → Hidden3(16) → Output(5)
- Implemented forward propagation with sigmoid and softmax activations
- Implemented backpropagation algorithm with gradient descent optimization
- Trained model on 80% training data, evaluated on 20% test data
- Visualized decision boundaries and training loss convergence

**Methodology:**
- One-hot encoding for multi-class labels
- Train-test split (80-20%)
- Mini-batch gradient descent with learning rate optimization
- Cross-entropy loss computation
- Sigmoid activation for hidden layers, Softmax for output layer

**Tools & Libraries Used:**
- Python, NumPy, Matplotlib
- Manual implementation (no TensorFlow/Keras)
- Custom gradient descent and backpropagation algorithms

### Practice

#### **Class Practice: CP**
[View Notebook](Mid/CP.ipynb)

---

## Final Term

### Assignments

#### **Assignment 1: Digit Recognition using Neural Network**
[View Notebook](Final/Assignment_1_Digit_Recognization_using_NN.ipynb)

**Task:** Build a Neural Network to recognize handwritten digits from MNIST dataset.

**What Was Done:**
- Loaded MNIST dataset (60,000 training images, 10,000 test images)
- Preprocessed images: normalized pixel values (0-1 range) and flattened 28×28 to 784D vectors
- Built Sequential Neural Network: Input(784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)
- Compiled with Adam optimizer and sparse categorical crossentropy loss
- Trained for 10 epochs with batch size 32
- Achieved high accuracy on digit classification task

**Methodology:**
- Image normalization and flattening for dense layer input
- Multi-layer perceptron architecture with ReLU activations
- Adam optimizer for efficient training
- Validation split during training (10%)
- Model saved as `mnist_model.h5`

**Tools & Libraries Used:**
- TensorFlow, Keras, NumPy, Matplotlib
- Pre-trained MNIST dataset from Keras
- Model architecture visualization

---

#### **Assignment 2: Face Recognition using CNN**
[View Notebook](Final/Assignment_2_Face_Recognization_CNN.ipynb)

**Task:** Implement Convolutional Neural Network for face recognition across 35 different people.

**What Was Done:**
- Loaded face dataset from 35 individuals (CVPR face dataset)
- Preprocessed images: resized to 128×128 pixels, normalized to 0-1 range
- Implemented custom CNN architecture with convolutional and pooling layers
- Data split: 80% training, 20% validation (stratified split)
- Trained CNN model for face classification
- Evaluated model performance on validation set
- Saved trained model as `face_model.keras`

**Methodology:**
- Image loading and preprocessing using OpenCV
- Stratified train-validation split to maintain class distribution
- Convolutional feature extraction with ReLU activation
- Max pooling for dimensionality reduction
- Fully connected layers for classification
- One-hot encoding of class labels

**Tools & Libraries Used:**
- TensorFlow, Keras, NumPy, OpenCV, Matplotlib
- Scikit-learn for train-test split
- Keras Sequential and Functional APIs
- Warning filters for clean output

---

#### **Assignment 3: Face Recognition using Transfer Learning**
[View Notebook](Final/Assignment_3_Face_Recognzation_Transfer_Learning.ipynb)

**Task:** Leverage transfer learning with pre-trained models for improved face recognition.

**What Was Done:**
- Loaded Haar Cascade for face detection from images
- Implemented face detection and cropping preprocessing pipeline
- Loaded and fine-tuned pre-trained deep learning model
- Applied transfer learning on CVPR face dataset (128×128 images)
- Detected, cropped, and resized faces automatically
- Trained model with class names stored in `class_names_student.json`
- Achieved improved accuracy using transfer learning approach

**Methodology:**
- Haar Cascade face detection for automatic face cropping
- Transfer learning: leveraged pre-trained model weights
- Fine-tuning on target dataset (face recognition task)
- Data augmentation and preprocessing
- Fallback to full image if face not detected
- Seaborn and Matplotlib for visualization

**Tools & Libraries Used:**
- TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn
- Haar Cascade classifier for face detection
- Transfer learning with pre-trained deep networks
- JSON for storing class metadata

---

### Demos

#### **Digit Recognition Demo**
[View Notebook](Final/Digit_demo.ipynb)

**Description:** Interactive demonstration of the trained MNIST digit recognition model. Users can test predictions on sample images or new data.

---

#### **Face Recognition Demo**
[View Notebook](Final/Face_demo.ipynb)

**Description:** Interactive demonstration of the trained face recognition models. Shows predictions and confidence scores for face classification.

---

## Key Technologies & Frameworks

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV (cv2)
- **Data Processing:** NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Custom Implementations:** KNN, Neural Networks, Backpropagation from scratch

## Data Files

- `mnist_model.h5` - Trained MNIST digit recognition model
- `face_model.keras` - Trained CNN face recognition model
- `class_names_student.json` - Class labels for face dataset
- `synthetic_data.csv` - Synthetic dataset for neural network practice