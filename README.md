# Automated Dental Restoration Detection & Human Identification via Deep Learning

This repository contains the end-to-end computer vision and machine learning pipeline developed for my MSc thesis at Istanbul Medeniyet University. The project automates the process of identifying individuals by classifying dental restorations from panoramic X-ray images.

## Project Architecture
The system operates in a two-stage pipeline:
1. **Tooth Segmentation & Numbering:** Ingests panoramic radiographs and segments individual teeth using an existing Mask R-CNN module, automatically labeling them via the FDI World Dental Federation notation system.
2. **Restoration Classification (My Core Work):** A custom 3-layer Convolutional Neural Network (CNN) built in TensorFlow/Keras that classifies each tooth crop into 5 categories: Composite Fillings, Implants, Endodontic Treatments, Non-restored, and Orthodontic Wires.
3. **Biometric E-Matching:** Generates 32-dimensional chronological feature vectors ($l_{12x}$) to model patient dental variations over time and matches profiles using Random Forest and SVM/SMO architectures.

## Key Performance Results
* **Deep Learning Classifier:** Achieved a **94% validation accuracy** across 3,013 dental images, with a **100% precision rate for implant detection**.
* **SVM / SMO Matching:** Secured **96.19% accuracy** in biometric profile matching.
* **Random Forest Matching:** Achieved **94.27% accuracy**, vastly outperforming traditional baseline distance metrics (Hamming, Jaccard, Cosine).

## Repository Structure
* `/models` - Custom 3-layer CNN model definition in TensorFlow.
* `/notebooks` - Google Colab notebooks used for data augmentation and training loops.
* `/plots` - Confusion matrices and accuracy/loss curves.

## Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Matplotlib
