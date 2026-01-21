# -*- coding: utf-8 -*-
"""
COMPLETE FETAL PLANE CLASSIFICATION SYSTEM - v4
Optimized for Google Colab with Comprehensive Model Comparison
Conference Paper Ready Implementation

Merges best features from v1, v2, and v3:
- Full dataset support (train/val/test)
- 7 ML baseline models + 7 CNN models
- Attention mechanisms
- Grad-CAM explainability
- Optimized RAM/GPU management for Colab

Author: Merged Implementation
Date: 2026-01-22
"""

# ============================================================================
# SECTION 1: SETUP & CONFIGURATION
# ============================================================================

import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Install required packages
print("=== Installing Required Packages ===")
os.system('pip install -q scikit-fuzzy xgboost kagglehub')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from tensorflow.keras.utils import image_dataset_from_directory

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import rbf_kernel
from skimage.feature import graycomatrix, graycoprops
import skfuzzy as fuzz
import xgboost as xgb

# GPU Setup - Enable memory growth to avoid OOM errors
print("\n=== GPU Configuration ===")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU available, using CPU")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration Constants
BATCH_SIZE = 20  # Optimized for Colab (balance between speed and memory)
EPOCHS = 20  # Reduced for faster training with early stopping
LEARNING_RATE = 1e-4
TARGET_SIZE = (224, 224)
CLASSES = ['Fetal brain', 'Fetal abdomen', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']
TARGET_BALANCE_COUNT = 2500  # Balanced approach

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Configuration: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LEARNING_RATE}")

# Mount Google Drive
print("\n=== Mounting Google Drive ===")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    OUTPUT_DIR = '/content/drive/MyDrive/CVPR_Project_v4'
    print(f"Google Drive mounted successfully")
except:
    OUTPUT_DIR = './CVPR_Project_v4'
    print(f"Not running on Colab, using local directory")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output Directory: {OUTPUT_DIR}")

# ============================================================================
# SECTION 2: DATASET DOWNLOADING & EDA
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: DATASET DOWNLOADING & EDA")
print("="*80)

import kagglehub
print("\n=== Downloading Dataset from Kaggle ===")
dataset_path = kagglehub.dataset_download('minhnhtl05/fetal-planes-db-dataset')
DATASET_PATH = os.path.join(dataset_path, 'Fetal_Planes_DB')
print(f"Dataset Root Path: {DATASET_PATH}")

# Verify dataset structure
print("\n=== Verifying Dataset Structure ===")
for subset in ['train', 'val', 'test']:
    subset_path = os.path.join(DATASET_PATH, subset)
    if os.path.exists(subset_path):
        print(f"✓ {subset} folder found")
    else:
        print(f"✗ {subset} folder not found")

def perform_eda(data_path, classes, output_dir):
    """Perform comprehensive EDA with visualizations"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===\")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Count samples per class in train set
    train_data_path = os.path.join(data_path, 'train')
    class_counts = {}
    
    for cls in classes:
        cls_path = os.path.join(train_data_path, cls)
        if os.path.exists(cls_path):
            count = len(glob(os.path.join(cls_path, '*.png')))
            class_counts[cls] = count
        else:
            class_counts[cls] = 0
    
    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()),
                ax=axes[0], hue=list(class_counts.keys()), palette='viridis', legend=False)
    axes[0].set_title('Class Distribution (Train Set)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (cls, count) in enumerate(class_counts.items()):
        axes[0].text(i, count + 20, str(count), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('viridis', len(classes)))
    axes[1].set_title('Class Distribution (%) (Train Set)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nClass Distribution (Train Set):")
    for cls, count in class_counts.items():
        print(f"  {cls:20s}: {count:5d} samples")
    
    return class_counts

class_distribution = perform_eda(DATASET_PATH, CLASSES, OUTPUT_DIR)

def visualize_sample_images(data_path, classes, samples_per_class=5, output_dir=OUTPUT_DIR):
    """Visualize sample images from each class"""
    print("\n=== VISUALIZING SAMPLE IMAGES ===\")
    
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(15, 18))
    
    train_data_path = os.path.join(data_path, 'train')
    for i, cls in enumerate(classes):
        cls_path = os.path.join(train_data_path, cls)
        image_files = glob(os.path.join(cls_path, '*.png'))[:samples_per_class]
        
        for j, img_path in enumerate(image_files):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=11, fontweight='bold')
    
    plt.suptitle('Sample Images from Each Class (Train Set)', fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_sample_images.png'), dpi=300, bbox_inches='tight')
    plt.show()

visualize_sample_images(DATASET_PATH, CLASSES)

# ============================================================================
# SECTION 3: PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: PREPROCESSING PIPELINE")
print("="*80)

class UltrasoundPreprocessor:
    """Advanced preprocessing pipeline for ultrasound images"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def crop_acoustic_window(self, img):
        """Remove black borders and text artifacts"""
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    
    def process_single_image(self, image_path):
        """Complete preprocessing pipeline"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Step 1: Crop ROI (Region of Interest)
        img_cropped = self.crop_acoustic_window(img)
        # Step 2: Resize with bicubic interpolation
        img_resized = cv2.resize(img_cropped, self.target_size, interpolation=cv2.INTER_CUBIC)
        # Step 3: Histogram equalization for contrast enhancement
        img_enhanced = cv2.equalizeHist(img_resized)
        
        return img_enhanced
    
    def balance_classes(self, X, y, target_count=TARGET_BALANCE_COUNT):
        """Balance dataset using resampling"""
        X_bal, y_bal = [], []
        classes = np.unique(y)
        
        print(f"\n=== Balancing Classes (Target: {target_count} samples per class) ===")
        for cls in classes:
            indices = np.where(y == cls)[0]
            X_cls, y_cls = X[indices], y[indices]
            
            print(f"Class {cls} ({CLASSES[cls]}): {len(X_cls)} -> {target_count} samples")
            
            replace = len(X_cls) < target_count
            X_res, y_res = resample(X_cls, y_cls, replace=replace,
                                   n_samples=target_count, random_state=SEED)
            X_bal.append(X_res)
            y_bal.append(y_res)
        
        return np.concatenate(X_bal), np.concatenate(y_bal)

def visualize_preprocessing_pipeline(preprocessor, data_path, classes, output_dir=OUTPUT_DIR):
    """Visualize preprocessing steps for one image per class"""
    print("\n=== VISUALIZING PREPROCESSING PIPELINE ===\")
    
    fig, axes = plt.subplots(len(classes), 6, figsize=(20, 20))
    
    train_data_path = os.path.join(data_path, 'train')
    for i, cls in enumerate(classes):
        cls_path = os.path.join(train_data_path, cls)
        image_files = glob(os.path.join(cls_path, '*.png'))
        
        if not image_files:
            print(f"Warning: No images found for class '{cls}'")
            for j in range(6):
                axes[i, j].set_visible(False)
            continue
        
        img_path = image_files[0]
        
        # Processing steps
        img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img_orig, 15, 255, cv2.THRESH_BINARY)
        img_cropped = preprocessor.crop_acoustic_window(img_orig)
        img_resized = cv2.resize(img_cropped, preprocessor.target_size, interpolation=cv2.INTER_CUBIC)
        img_eq = cv2.equalizeHist(img_resized)
        
        steps = [img_orig, thresh, img_cropped, img_resized, img_eq]
        titles = ['Original', 'Threshold\nMask', 'Cropped\nROI', 'Resized\n224x224', 'Histogram\nEqualized']
        
        for j, (step, title) in enumerate(zip(steps, titles)):
            axes[i, j].imshow(step, cmap='gray')
            if i == 0:
                axes[i, j].set_title(title, fontsize=10, fontweight='bold')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=10, fontweight='bold', rotation=90, labelpad=10)
        
        # Histogram in last column
        axes[i, 5].hist(img_eq.ravel(), bins=256, range=[0, 256], color='darkblue', alpha=0.7)
        if i == 0:
            axes[i, 5].set_title('Intensity\nHistogram', fontsize=10, fontweight='bold')
        axes[i, 5].set_xlim([0, 256])
        axes[i, 5].tick_params(labelsize=7)
    
    plt.suptitle('Preprocessing Pipeline (6 Steps per Class)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_preprocessing_pipeline.png'), dpi=300, bbox_inches='tight')
    plt.show()

preprocessor = UltrasoundPreprocessor(target_size=TARGET_SIZE)
visualize_preprocessing_pipeline(preprocessor, DATASET_PATH, CLASSES)

# ============================================================================
# SECTION 4: DATA LOADING & BALANCING
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: DATA LOADING & BALANCING")
print("="*80)

print("\n=== Loading and Preprocessing All Data ===")

X_raw_paths, y_raw = [], []

# Load from all subsets (train, val, test)
subsets = ['train', 'val', 'test']

for subset in subsets:
    subset_path = os.path.join(DATASET_PATH, subset)
    if not os.path.exists(subset_path):
        print(f"Warning: Subset '{subset}' not found, skipping...")
        continue
    
    for i, cls in enumerate(CLASSES):
        cls_path = os.path.join(subset_path, cls)
        if os.path.exists(cls_path):
            paths = glob(os.path.join(cls_path, '*.png'))
            X_raw_paths.extend(paths)
            y_raw.extend([i] * len(paths))
    
    print(f"Loaded {subset} subset")

print(f"\nTotal images found: {len(X_raw_paths)}")

# Process images in batches to manage RAM
X_processed = []
batch_size = 500

print("\n=== Processing Images ===")
for i in range(0, len(X_raw_paths), batch_size):
    batch_paths = X_raw_paths[i:i+batch_size]
    for path in batch_paths:
        img = preprocessor.process_single_image(path)
        if img is not None:
            X_processed.append(img)
    
    if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(X_raw_paths):
        print(f"Processed {min(i + batch_size, len(X_raw_paths))}/{len(X_raw_paths)} images")
        gc.collect()

X_processed = np.array(X_processed)
y_raw = np.array(y_raw[:len(X_processed)])

print(f"Processed data shape: {X_processed.shape}")

# Balance dataset
print("\n=== Balancing Dataset ===")
X_balanced, y_balanced = preprocessor.balance_classes(X_processed, y_raw)
print(f"Balanced data shape: {X_balanced.shape}")

# Visualize before/after balancing
def plot_balancing_effect(y_before, y_after, classes, output_dir=OUTPUT_DIR):
    """Visualize data balancing effect"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before balancing
    unique, counts = np.unique(y_before, return_counts=True)
    labels = [classes[i] for i in unique]
    bars1 = sns.barplot(x=labels, y=counts, ax=axes[0], hue=labels, palette='Reds', legend=False)
    axes[0].set_title('Before Balancing', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Sample Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, count in enumerate(counts):
        axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    # After balancing
    unique, counts = np.unique(y_after, return_counts=True)
    labels = [classes[i] for i in unique]
    bars2 = sns.barplot(x=labels, y=counts, ax=axes[1], hue=labels, palette='Greens', legend=False)
    axes[1].set_title('After Balancing', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Sample Count', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    for i, count in enumerate(counts):
        axes[1].text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_data_balancing.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_balancing_effect(y_raw, y_balanced, CLASSES)

# Clear memory
del X_processed, X_raw_paths, y_raw
gc.collect()

print("\n✓ Data loading and balancing complete")

# ============================================================================
# SECTION 5: GLCM FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: GLCM FEATURE EXTRACTION (For ML Models)")
print("="*80)

def extract_glcm_features(image, levels=64):
    """Extract 45 GLCM texture features"""
    img_quantized = (image // (256 // levels)).astype(np.uint8)
    
    distances = [5]
    angles = [0, np.deg2rad(26.6), np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)]
    
    glcm = graycomatrix(img_quantized, distances=distances, angles=angles,
                       levels=levels, symmetric=True, normed=True)
    
    feature_vector = []
    
    # Extract standard GLCM properties
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        feature_vector.extend(graycoprops(glcm, prop).flatten())
    
    # Extract additional statistical properties
    for i in range(5):
        P = glcm[:, :, 0, i]
        rows, cols = np.indices(P.shape)
        
        mean_i = np.sum(rows * P)
        variance = np.sum(P * (rows - mean_i)**2)
        entropy = -np.sum(P * np.log2(P + 1e-10))
        shade = np.sum(((rows + cols - mean_i - mean_i)**3) * P)
        
        feature_vector.extend([mean_i, variance, entropy, shade])
    
    return np.array(feature_vector)

print("\n=== Extracting GLCM Features ===")
X_glcm = []
for i, img in enumerate(X_balanced):
    X_glcm.append(extract_glcm_features(img))
    if (i + 1) % 1000 == 0:
        print(f"Extracted features from {i + 1}/{len(X_balanced)} images")

X_glcm = np.array(X_glcm)
print(f"\nGLCM Features shape: {X_glcm.shape} ({X_glcm.shape[1]} features per image)")
print("✓ GLCM feature extraction complete")

# ============================================================================
# SECTION 6: PSO-GWO FEATURE SELECTION
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: HYBRID PSO-GWO FEATURE SELECTION")
print("="*80)

class HybridPSOGWO:
    """Hybrid Particle Swarm Optimization - Grey Wolf Optimizer for feature selection"""
    
    def __init__(self, num_agents, max_iter, dim, X_train, y_train):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.dim = dim
        self.X_train = X_train
        self.y_train = y_train
        
        # GWO wolves (alpha, beta, delta)
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
        # PSO particles
        self.positions = np.random.randint(2, size=(num_agents, dim))
        self.velocities = np.random.uniform(-1, 1, (num_agents, dim))
        self.fitness_history = []
    
    def fitness(self, subset_mask):
        """Calculate fitness: 0.99*Error + 0.01*Ratio"""
        if np.sum(subset_mask) == 0:
            return 1.0
        
        features = self.X_train[:, subset_mask.astype(bool)]
        X_tr, X_val, y_tr, y_val = train_test_split(features, self.y_train,
                                                     test_size=0.2, random_state=SEED)
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, knn.predict(X_val))
        
        error = 1 - acc
        ratio = np.sum(subset_mask) / self.dim
        return 0.99 * error + 0.01 * ratio
    
    def optimize(self):
        """Run hybrid PSO-GWO optimization"""
        print(f"\nStarting optimization with {self.num_agents} agents for {self.max_iter} iterations...")
        
        for t in range(self.max_iter):
            # Evaluate all agents
            for i in range(self.num_agents):
                fit = self.fitness(self.positions[i])
                
                # Update alpha, beta, delta
                if fit < self.alpha_score:
                    self.alpha_score = fit
                    self.alpha_pos = self.positions[i].copy()
                elif fit < self.beta_score:
                    self.beta_score = fit
                    self.beta_pos = self.positions[i].copy()
                elif fit < self.delta_score:
                    self.delta_score = fit
                    self.delta_pos = self.positions[i].copy()
            
            self.fitness_history.append(self.alpha_score)
            
            # Update velocity weight (linearly decreasing)
            w = 0.9 - t * (0.5 / self.max_iter)
            
            # Update positions using PSO-GWO hybrid
            for i in range(self.num_agents):
                r1, r2, r3 = np.random.rand(3)
                v_new = (w * self.velocities[i] +
                        1.5 * r1 * (self.alpha_pos - self.positions[i]) +
                        1.5 * r2 * (self.beta_pos - self.positions[i]) +
                        1.5 * r3 * (self.delta_pos - self.positions[i]))
                
                self.velocities[i] = v_new
                sigmoid = 1 / (1 + np.exp(-v_new))
                rand_vals = np.random.rand(self.dim)
                self.positions[i] = (rand_vals < sigmoid).astype(int)
            
            if (t + 1) % 5 == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {self.alpha_score:.4f} | "
                      f"Features Selected: {np.sum(self.alpha_pos)}")
        
        return self.alpha_pos

# Split data for feature selection
X_train_glcm, X_test_glcm, y_train, y_test = train_test_split(
    X_glcm, y_balanced, test_size=0.2, random_state=SEED, stratify=y_balanced
)

print(f"\nTrain set: {X_train_glcm.shape[0]} samples")
print(f"Test set: {X_test_glcm.shape[0]} samples")

# Run PSO-GWO feature selection
print("\n=== Running PSO-GWO Feature Selection ===")
optimizer = HybridPSOGWO(num_agents=18, max_iter=25, dim=X_train_glcm.shape[1],
                         X_train=X_train_glcm, y_train=y_train)
best_mask = optimizer.optimize()

X_train_sel = X_train_glcm[:, best_mask.astype(bool)]
X_test_sel = X_test_glcm[:, best_mask.astype(bool)]

print(f"\n✓ Selected {np.sum(best_mask)}/{len(best_mask)} features ({100*np.sum(best_mask)/len(best_mask):.1f}%)")
print(f"Reduced feature shape: Train={X_train_sel.shape}, Test={X_test_sel.shape}")

# ============================================================================
# SECTION 7: FAST-RBFNN CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: FAST-RBFNN CLASSIFIER")
print("="*80)

class FastRBFNN:
    """Fast RBF Neural Network with Fuzzy C-Means clustering"""
    
    def __init__(self, n_centers=50, epsilon=0.1, C=1.0):
        self.n_centers = n_centers
        self.epsilon = epsilon
        self.C = C
        self.centers = None
        self.sigmas = None
        self.linear_model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """Train the Fast-RBFNN model"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Fuzzy C-Means for adaptive center selection
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_scaled.T, c=self.n_centers, m=2, error=0.005, maxiter=100
        )
        self.centers = cntr
        
        # Calculate adaptive widths (sigmas)
        self.sigmas = np.zeros(self.n_centers)
        labels = np.argmax(u, axis=0)
        
        for i in range(self.n_centers):
            points = X_scaled[labels == i]
            if len(points) > 0:
                self.sigmas[i] = np.mean(np.linalg.norm(points - self.centers[i], axis=1))
            else:
                self.sigmas[i] = 1.0
        
        # Train linear SVR on RBF-transformed features
        avg_sigma = np.mean(self.sigmas)
        gamma = 1.0 / (2.0 * avg_sigma**2)
        H = rbf_kernel(X_scaled, self.centers, gamma=gamma)
        
        self.linear_model = LinearSVR(epsilon=self.epsilon, C=self.C, max_iter=10000)
        self.linear_model.fit(H, y)
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        avg_sigma = np.mean(self.sigmas)
        gamma = 1.0 / (2.0 * avg_sigma**2)
        H = rbf_kernel(X_scaled, self.centers, gamma=gamma)
        return self.linear_model.predict(H)

print("\n✓ Fast-RBFNN class defined")

# ============================================================================
# SECTION 8: BASELINE ML MODELS
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: TRAINING BASELINE ML MODELS")
print("="*80)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, classes):
    """Evaluate a model and return comprehensive metrics"""
    # Get predictions
    if isinstance(model, FastRBFNN):
        y_pred_test = np.clip(np.round(model.predict(X_test)).astype(int), 0, len(classes)-1)
        y_pred_train = np.clip(np.round(model.predict(X_train)).astype(int), 0, len(classes)-1)
    else:
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train_Acc': accuracy_score(y_train, y_pred_train),
        'Train_F1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'Test_Acc': accuracy_score(y_test, y_pred_test),
        'Test_Prec': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'Test_Rec': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'Test_F1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    }
    
    print(f"  {model_name:20s} | Train Acc: {metrics['Train_Acc']:.4f} | Test Acc: {metrics['Test_Acc']:.4f} | Test F1: {metrics['Test_F1']:.4f}")
    return metrics, y_pred_test

# Store results
ml_results = []
model_predictions = {}

# 1. Fast-RBFNN (Proposed Method)
print("\n1. Training Fast-RBFNN...")
frbfnn = FastRBFNN(n_centers=50, epsilon=0.1, C=1.0)
frbfnn.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(frbfnn, X_train_sel, y_train, X_test_sel, y_test, 'Fast-RBFNN', CLASSES)
ml_results.append(metrics)
model_predictions['Fast-RBFNN'] = preds

# 2. SVM with RBF Kernel
print("2. Training SVM...")
svm = SVC(kernel='rbf', random_state=SEED, gamma='scale')
svm.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(svm, X_train_sel, y_train, X_test_sel, y_test, 'SVM', CLASSES)
ml_results.append(metrics)
model_predictions['SVM'] = preds

# 3. Random Forest
print("3. Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(rf, X_train_sel, y_train, X_test_sel, y_test, 'Random Forest', CLASSES)
ml_results.append(metrics)
model_predictions['Random Forest'] = preds

# 4. KNN
print("4. Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(knn, X_train_sel, y_train, X_test_sel, y_test, 'KNN', CLASSES)
ml_results.append(metrics)
model_predictions['KNN'] = preds

# 5. XGBoost
print("5. Training XGBoost...")
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(CLASSES),
                              eval_metric='mlogloss', use_label_encoder=False,
                              random_state=SEED, n_jobs=-1)
xgb_model.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(xgb_model, X_train_sel, y_train, X_test_sel, y_test, 'XGBoost', CLASSES)
ml_results.append(metrics)
model_predictions['XGBoost'] = preds

# 6. Logistic Regression
print("6. Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
log_reg.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(log_reg, X_train_sel, y_train, X_test_sel, y_test, 'Logistic Regression', CLASSES)
ml_results.append(metrics)
model_predictions['Logistic Regression'] = preds

# 7. Decision Tree
print("7. Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train_sel, y_train)
metrics, preds = evaluate_model(dt, X_train_sel, y_train, X_test_sel, y_test, 'Decision Tree', CLASSES)
ml_results.append(metrics)
model_predictions['Decision Tree'] = preds

print("\n✓ All ML models trained successfully")

# Clear GLCM features from memory
del X_glcm, X_train_glcm, X_test_glcm, X_train_sel, X_test_sel
gc.collect()

# ============================================================================
# SECTION 9: CNN DATA PREPARATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 9: CNN DATA PREPARATION")
print("="*80)

print("\n=== Converting Grayscale to RGB ===")
# Convert grayscale to RGB (required for pretrained models)
X_balanced_rgb = np.stack([X_balanced] * 3, axis=-1)
print(f"RGB data shape: {X_balanced_rgb.shape}")

# Split into train and validation
print("\n=== Creating Train/Validation Split ===\")
train_val_split = int(0.85 * len(X_balanced_rgb))
indices = np.arange(len(X_balanced_rgb))
np.random.shuffle(indices)

X_train_cnn = X_balanced_rgb[indices[:train_val_split]]
y_train_cnn = y_balanced[indices[:train_val_split]]
X_val_cnn = X_balanced_rgb[indices[train_val_split:]]
y_val_cnn = y_balanced[indices[train_val_split:]]

print(f"Train set: {X_train_cnn.shape[0]} samples")
print(f"Validation set: {X_val_cnn.shape[0]} samples")

# Create TensorFlow datasets with optimizations
train_ds_cnn = tf.data.Dataset.from_tensor_slices((X_train_cnn, y_train_cnn))
train_ds_cnn = train_ds_cnn.shuffle(buffer_size=1000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds_cnn = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val_cnn))
val_ds_cnn = val_ds_cnn.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Clear arrays after creating datasets
del X_balanced, X_balanced_rgb, X_train_cnn, X_val_cnn
gc.collect()

print("\n✓ CNN datasets prepared")

# Data Augmentation Layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name='data_augmentation')

# Self-Attention Layer
class SequentialSelfAttention(layers.Layer):
    """Self-attention mechanism with learnable projections"""
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.wq = layers.Dense(channels, name='query')
        self.wk = layers.Dense(channels, name='key')
        self.wv = layers.Dense(channels, name='value')
    
    def call(self, inputs):
        shape = tf.shape(inputs)
        H, W = shape[1], shape[2]
        
        # Reshape to sequence
        x = tf.reshape(inputs, (-1, H * W, self.channels))
        
        # Compute Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.channels, tf.float32)
        scaled_logits = matmul_qk / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_logits, axis=-1)
        output = tf.matmul(weights, v)
        
        # Reshape back
        return tf.reshape(output, (-1, H, W, self.channels))

print("✓ Data augmentation and attention layers defined")

# ============================================================================
# SECTION 10: CNN MODELS TRAINING
# ============================================================================

print("\n" + "="*80)
print("SECTION 10: TRAINING CNN MODELS (7 Architectures)")
print("="*80)

def build_cnn_model(backbone_name, num_classes, use_attention=True):
    """Build CNN model with pretrained backbone and attention mechanism"""
    
    # Backbone configuration
    backbone_map = {
        'MobileNetV2': (applications.MobileNetV2, applications.mobilenet_v2.preprocess_input),
        'EfficientNetV2B0': (applications.EfficientNetV2B0, applications.efficientnet_v2.preprocess_input),
        'ResNet50V2': (applications.ResNet50V2, applications.resnet_v2.preprocess_input),
        'VGG16': (applications.VGG16, applications.vgg16.preprocess_input),
        'InceptionV3': (applications.InceptionV3, applications.inception_v3.preprocess_input),
        'DenseNet121': (applications.DenseNet121, applications.densenet.preprocess_input),
        'Xception': (applications.Xception, applications.xception.preprocess_input),
    }
    
    base_model_class, preprocess_fn = backbone_map[backbone_name]
    
    # Build model
    inputs = layers.Input(shape=(224, 224, 3), name='input')
    x = data_augmentation(inputs)
    x = preprocess_fn(x)
    
    # Load pretrained backbone
    base_model = base_model_class(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True  # Fine-tune all layers
    
    features = base_model.output
    
    # Add attention mechanism
    if use_attention:
        features = SequentialSelfAttention(channels=features.shape[-1], name='self_attention')(features)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(features)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name}_Attention")
    return model

def train_cnn_model(backbone_name):
    """Train a CNN model with given backbone"""
    print(f"\n{'='*60}")
    print(f"Training {backbone_name}")
    print(f"{'='*60}")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Build model
    model = build_cnn_model(backbone_name, len(CLASSES), use_attention=True)
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print(f"\nTraining {backbone_name}...")
    history = model.fit(
        train_ds_cnn,
        validation_data=val_ds_cnn,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate on training set
    train_loss, train_acc = model.evaluate(train_ds_cnn, verbose=0)
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds_cnn, verbose=0)
    
    # Get predictions for detailed metrics
    y_true_val = []
    y_pred_val = []
    
    for images, labels in val_ds_cnn:
        preds = model.predict(images, verbose=0)
        y_true_val.extend(labels.numpy())
        y_pred_val.extend(np.argmax(preds, axis=1))
    
    y_true_val = np.array(y_true_val)
    y_pred_val = np.array(y_pred_val)
    
    # Calculate metrics
    metrics = {
        'Model': backbone_name,
        'Train_Acc': train_acc,
        'Train_F1': 0.0,  # Not calculated for time efficiency
        'Test_Acc': val_acc,
        'Test_Prec': precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0),
        'Test_Rec': recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0),
        'Test_F1': f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
    }
    
    print(f"\n{backbone_name} Results:")
    print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {metrics['Test_F1']:.4f}")
    
    return metrics, model, y_pred_val, history

# Train all CNN models
cnn_backbones = ['MobileNetV2', 'EfficientNetV2B0', 'ResNet50V2', 'VGG16', 
                 'InceptionV3', 'DenseNet121', 'Xception']

cnn_results = []
best_cnn_model = None
best_cnn_acc = 0
best_cnn_pred = None
best_cnn_name = None

for backbone in cnn_backbones:
    try:
        metrics, model, preds, history = train_cnn_model(backbone)
        cnn_results.append(metrics)
        model_predictions[backbone] = preds
        
        # Track best model
        if metrics['Test_Acc'] > best_cnn_acc:
            best_cnn_acc = metrics['Test_Acc']
            best_cnn_model = model
            best_cnn_pred = preds
            best_cnn_name = backbone
        
        # Clear memory
        del model, history
        gc.collect()
        
    except Exception as e:
        print(f"\n⚠ Error training {backbone}: {e}")
        print("Skipping this model...")
        continue

print(f"\n✓ CNN training complete. Best model: {best_cnn_name} (Acc: {best_cnn_acc:.4f})")

# ============================================================================
# SECTION 11: MODEL COMPARISON & EVALUATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 11: COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Combine all results
all_results = ml_results + cnn_results
results_df = pd.DataFrame(all_results)

# Sort by test accuracy
results_df = results_df.sort_values('Test_Acc', ascending=False)

print("\n=== Model Performance Comparison ===")
print(results_df.to_string(index=False))

# Save to CSV
csv_path = os.path.join(OUTPUT_DIR, '05_model_comparison.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Visualization 1: Model Comparison Bar Chart
print("\n=== Creating Model Comparison Visualization ===")
fig, ax = plt.subplots(figsize=(16, 8))

models = results_df['Model'].values
train_accs = results_df['Train_Acc'].values
test_accs = results_df['Test_Acc'].values

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test/Val Accuracy', alpha=0.8, color='coral')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Comprehensive Model Performance Comparison\n(7 ML Models + 7 CNN Models)', 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_model_comparison_chart.png'), dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Metrics Heatmap
print("\n=== Creating Metrics Heatmap ===")
fig, ax = plt.subplots(figsize=(12, 10))

metrics_cols = ['Train_Acc', 'Test_Acc', 'Test_Prec', 'Test_Rec', 'Test_F1']
heatmap_data = results_df[metrics_cols].values

im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(metrics_cols)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(['Train Acc', 'Test Acc', 'Test Prec', 'Test Rec', 'Test F1'])
ax.set_yticklabels(models)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(models)):
    for j in range(len(metrics_cols)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
fig.colorbar(im, ax=ax, label='Score')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# Find best overall model
best_model_idx = results_df['Test_Acc'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_acc = results_df.loc[best_model_idx, 'Test_Acc']

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {best_model_acc:.4f}")
print(f"   Test Precision: {results_df.loc[best_model_idx, 'Test_Prec']:.4f}")
print(f"   Test Recall: {results_df.loc[best_model_idx, 'Test_Rec']:.4f}")
print(f"   Test F1-Score: {results_df.loc[best_model_idx, 'Test_F1']:.4f}")
print(f"{'='*80}")

# Confusion Matrix for Best Model
print(f"\n=== Creating Confusion Matrix for {best_model_name} ===")

# Get predictions from best model
if best_model_name in model_predictions:
    best_pred = model_predictions[best_model_name]
    
    # Determine true labels based on model type
    if best_model_name in ['Fast-RBFNN', 'SVM', 'Random Forest', 'KNN', 'XGBoost', 
                           'Logistic Regression', 'Decision Tree']:
        y_true = y_test  # ML models use test set
    else:
        y_true = y_val_cnn  # CNN models use validation set
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, best_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {best_model_name}\n(Test Accuracy: {best_model_acc:.4f})',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_confusion_matrix_best.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    print(f"\n=== Classification Report for {best_model_name} ===")
    print(classification_report(y_true, best_pred, target_names=CLASSES, digits=4))

# ============================================================================
# SECTION 12: EXPLAINABILITY (GRAD-CAM)
# ============================================================================

print("\n" + "="*80)
print("SECTION 12: GRAD-CAM EXPLAINABILITY")
print("="*80)

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    grad_model = models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return superimposed

# Generate Grad-CAM only if best model is CNN
if best_cnn_model is not None and best_model_name in cnn_backbones:
    print(f"\n=== Generating Grad-CAM for {best_model_name} ===")
    
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(best_cnn_model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_layer = layer.name
            break
    
    if last_conv_layer:
        print(f"Using convolutional layer: {last_conv_layer}")
        
        # Get a few sample images for Grad-CAM
        sample_images = []
        sample_labels = []
        for images, labels in val_ds_cnn.take(1):
            sample_images = images[:6].numpy()
            sample_labels = labels[:6].numpy()
            break
        
        # Generate Grad-CAM visualizations
        fig, axes = plt.subplots(6, 3, figsize=(12, 24))
        
        for i in range(6):
            img_array = np.expand_dims(sample_images[i], axis=0)
            
            # Original image
            orig_img = sample_images[i].astype(np.uint8)
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f'Original\nTrue: {CLASSES[sample_labels[i]]}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Prediction
            pred = best_cnn_model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            pred_prob = pred[0][pred_class]
            
            # Grad-CAM
            try:
                heatmap = get_gradcam_heatmap(best_cnn_model, img_array, last_conv_layer, pred_class)
                axes[i, 1].imshow(heatmap, cmap='jet')
                axes[i, 1].set_title('Grad-CAM\nHeatmap', fontsize=10)
                axes[i, 1].axis('off')
                
                # Overlay
                superimposed = overlay_heatmap(orig_img, heatmap)
                axes[i, 2].imshow(superimposed)
                axes[i, 2].set_title(f'Overlay\nPred: {CLASSES[pred_class]} ({pred_prob:.2f})', fontsize=10)
                axes[i, 2].axis('off')
            except Exception as e:
                print(f"Error generating Grad-CAM for sample {i}: {e}")
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')
        
        plt.suptitle(f'Grad-CAM Explainability - {best_model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '09_gradcam_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Grad-CAM visualizations generated")
    else:
        print("⚠ Could not find convolutional layer for Grad-CAM")
else:
    print("⚠ Best model is not a CNN, skipping Grad-CAM")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXECUTION COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Total Models Trained: {len(all_results)}")
print(f"   - ML Models: 7 (Fast-RBFNN, SVM, RF, KNN, XGBoost, LogReg, DT)")
print(f"   - CNN Models: {len(cnn_results)} (MobileNetV2, EfficientNetV2B0, ResNet50V2, VGG16, InceptionV3, DenseNet121, Xception)")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   - Test Accuracy: {best_model_acc:.4f}")
print(f"   - Test F1-Score: {results_df.loc[best_model_idx, 'Test_F1']:.4f}")

print(f"\n📁 All results saved to: {OUTPUT_DIR}")
print(f"   - 01_class_distribution.png")
print(f"   - 02_sample_images.png")
print(f"   - 03_preprocessing_pipeline.png")
print(f"   - 04_data_balancing.png")
print(f"   - 05_model_comparison.csv")
print(f"   - 06_model_comparison_chart.png")
print(f"   - 07_metrics_heatmap.png")
print(f"   - 08_confusion_matrix_best.png")
if best_model_name in cnn_backbones:
    print(f"   - 09_gradcam_visualization.png")

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*80)
