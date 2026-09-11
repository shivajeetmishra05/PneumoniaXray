# Chest X-Ray Pneumonia Classifier

A lightweight, containerized deep learning web application built with **Streamlit** and **TensorFlow**. This project utilizes a Convolutional Neural Network (CNN) trained on chest X-ray images to predict and classify the presence of **Pneumonia** versus **Normal** clinical cases.

Live App link: *(https://pneumoniaxray-bzgqk2yxm5jv8r35nvzkfj.streamlit.app/)*

---

## Dataset & Model Background
The underlying model was trained using the **Chest X-Ray Images (Pneumonia)** dataset, analyzing structural features in single-channel grayscale data.
* **Input Resolution:** 224x224 pixels (Grayscale)
* **Model Size:** ~127 MB (Tracked via Git LFS)
* **Performance:** Reached ~0.94 ROC AUC score during evaluations with an optimized classification threshold set to `0.50` to balance recall metrics.

---

## Project Structure
```text
├── .gitattributes          # Git LFS tracking configuration for large binaries
├── app.py                  # Streamlit application UI & predictive pipeline logic
├── chest_xray_model.keras  # Trained TensorFlow model weights (127 MB)
└── requirements.txt        # Production dependencies list
```

---

## Local Deployment Setup

If you want to run this application locally on your desktop or laptop computer, follow these sequential steps:

### 1. Prerequisites (Install Git LFS)
Because the model weights exceed GitHub's 100MB tracking limit, make sure **Git LFS** is configured on your system before cloning:
* **macOS:** `brew install git-lfs`
* **Windows/Linux:** Download the installer binaries via [git-lfs.com](https://git-lfs.com).

Once installed, initialize it globally:
```bash
git lfs install
```

### 2. Clone and Setup Environment
Clone the repository and move into your project root:
```bash
git clone https://github.com
cd YOUR_REPO_NAME

# Pull the physical model weights down from LFS pointers
git lfs pull
```

Create a virtual environment and load dependencies:
```bash
# Set up a stable environment (Python 3.11 or 3.12 recommended)
python -bin venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Execution
Fire up the local host web server loop:
```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment Notes

When hosting this platform on **Streamlit Cloud**, pay attention to the following system configurations:

* **Python Version Constraint:** You **must** manually toggle your Streamlit Advanced App Settings to **Python 3.11** or **Python 3.12**. The default Python 3.14 build environment does not support pre-compiled wheels for TensorFlow yet.
* **Git LFS Integration:** Streamlit Cloud natively checks for `.gitattributes` and pulls down the complete 127 MB weight file seamlessly upon container build cycles.

---

## Disclaimer
This web dashboard is intended solely for educational, research, and prototyping demonstration purposes. It does not provide actionable medical advice and must never substitute for a formal diagnosis or evaluation from a certified medical practitioner.
