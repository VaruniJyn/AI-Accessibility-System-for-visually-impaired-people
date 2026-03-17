# 🧠 AI Accessibility System for Visually Impaired People

## 📌 Overview

This project is an AI-powered system designed to help visually impaired people interact with their surroundings more easily. It uses computer vision and audio feedback to detect objects, read text, and provide real-time assistance.

The goal is to make everyday tasks like navigating spaces, identifying objects, and reading text more accessible using AI.

---

## 🚀 Features

* 🔍 **Object Detection** – Detects real-world objects using a camera
* 🗣️ **Text-to-Speech (TTS)** – Converts detected information into voice output
* 📷 **Real-time Camera Processing** – Works with live video input
* 🧾 **Text Recognition (OCR)** – Reads printed text from images
* ⚡ **Fast Processing** – Provides quick responses for better usability

---

## 🛠️ Tech Stack

### 👨‍💻 Programming Language

* Python

### 🤖 AI / ML

* TensorFlow
* PyTorch
* YOLO (You Only Look Once) for object detection

### 👁️ Computer Vision

* OpenCV

### 🔊 Audio Processing

* pyttsx3 (Text-to-Speech)

### 📚 OCR (Text Recognition)

* pytesseract

---

## 📦 Libraries Used

* opencv-python
* torch
* torchvision
* numpy
* pytesseract
* pyttsx3
* Pillow

---

## 🏗️ Project Structure

```
AI-Accessibility-System/
│── main.py                # Main application file
│── object_detection/      # YOLO model and detection logic
│── ocr/                   # Text recognition module
│── audio/                 # Text-to-speech functionality
│── utils/                 # Helper functions
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/AI-Accessibility-System.git
cd AI-Accessibility-System
```

2. Create virtual environment:

```
python -m venv venv
```

3. Activate environment:

* Windows:

```
venv\Scripts\activate
```

4. Install dependencies:

```
pip install -r requirements.txt
```

---

## 💡 How It Works (Simple Explanation)

1. The camera captures live video
2. AI model detects objects in the frame
3. OCR extracts text if present
4. The system converts results into speech
5. User hears the output in real-time

---

## 🎯 Use Cases

* Helping visually impaired people navigate surroundings
* Reading labels, signs, and documents
* Identifying objects in daily life
* Assistive technology for independence

---

## ⚠️ Limitations

* Requires good lighting for better accuracy
* OCR accuracy depends on text clarity
* Performance depends on system hardware

---



Give it a ⭐ on GitHub!
