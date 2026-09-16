# QNX Driver Drowsiness & Attention Monitoring

A real-time driver monitoring system designed to detect **driver drowsiness and reduced attention** using computer vision, machine learning, and **QNX Neutrino RTOS** on Raspberry Pi.

The project focuses on building a lightweight ML pipeline that can be integrated into a real-time operating system with bounded latency and task prioritization.

---

## Project Overview

Driver drowsiness is a major road-safety concern, particularly during long-duration driving and low-attention conditions.

This project aims to develop a real-time monitoring pipeline that processes camera frames, extracts driver-related visual features, performs drowsiness classification, and generates an alert when a sleepy state is detected.

The system is designed around the following pipeline:

```text
Camera
   ↓
Frame Capture
   ↓
Face / Eye Detection
   ↓
Image Preprocessing
   ↓
CNN-Based Drowsiness Classification
   ↓
Decision Task
   ↓
Drowsiness Alert
```

The final implementation is intended to run on **Raspberry Pi with QNX Neutrino RTOS**, with emphasis on real-time task scheduling and predictable response latency.

---

##  Objectives

* Detect whether the driver is **awake or sleepy**.
* Develop a lightweight CNN suitable for edge deployment.
* Export the trained model to **ONNX** format.
* Integrate the ML inference pipeline with QNX RTOS.
* Design real-time processing using prioritized tasks.
* Measure inference latency and processing performance.
* Generate an alert when drowsiness is detected.
* Build a modular architecture that can later incorporate additional driver-attention features.

---

##  Machine Learning

### Classification Classes

The current ML model performs binary classification:

| Class    | Description                         |
| -------- | ----------------------------------- |
| `awake`  | Driver appears alert                |
| `sleepy` | Driver exhibits signs of drowsiness |

### ML Pipeline

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train / Validation / Test Split
   ↓
CNN Training
   ↓
Model Evaluation
   ↓
ONNX Export
   ↓
Edge Deployment
```



### Model Output

```text
awake
sleepy
```

### Model Format

The trained model is available in ONNX format:

```text
models/drowsiness_cnn.onnx
```

The model is tracked using **Git LFS** because of its file size.

---

## 📊 Dataset

The project uses a labeled eye-state/drowsiness image dataset containing two categories:

* Awake
* Sleepy

The dataset is used for model training, validation, and testing.

The complete dataset is **not stored in this repository** because of its size.

---

## 💻 Technology Stack

### Machine Learning

* Python
* PyTorch
* OpenCV
* ONNX

### Computer Vision

* OpenCV
* Haar Cascade classifiers
* Facial landmark detection

### Embedded / Hardware

* Raspberry Pi 5
* Camera module / compatible camera input

### Real-Time Operating System

* QNX Neutrino RTOS
* QNX SDP

### Development

* Git
* GitHub
* Git LFS
* Visual Studio Code

---

## ⚙️ Real-Time QNX Architecture

The target system is structured as a collection of real-time tasks.

```text
                    ┌───────────────────┐
                    │      Camera       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Capture Task     │
                    └─────────┬─────────┘
                              │
                         Shared Buffer
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature / ML Task │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Decision Task    │
                    │   High Priority   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    Alert Task     │
                    └───────────────────┘
```

The QNX implementation is intended to use:

* Task prioritization
* Deadline-aware scheduling
* CPU budgeting
* Shared buffers
* Message queues
* Task pipelining

The **Decision Task** is designed to receive higher priority because timely drowsiness decisions are critical to the system.

---

## 📈 Performance Evaluation

The final system will be evaluated using real-time metrics such as:

* Classification accuracy
* Precision
* Recall
* F1-score
* Frames per second (FPS)
* Frame-processing latency
* End-to-end decision latency
* Alert response time
* CPU utilization
* Deadline compliance

These measurements will be collected after integration with the Raspberry Pi and QNX environment.

---


