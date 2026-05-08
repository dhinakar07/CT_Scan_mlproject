# 🫁 AI-Powered Pneumonia Detection & Automated Report Generation

A production-grade deep learning application that detects pneumonia from chest X-ray images using a hybrid **ResNet50 + CLIP** model, generates detailed medical reports using **Google Gemini AI**, and automatically emails the report as a PDF to the patient.

---

## 🚀 Demo

Upload a chest X-ray → Get an AI diagnosis → Receive a detailed PDF medical report in your email — all in under 60 seconds.

---

## 🧠 How It Works

1. **Upload** a chest X-ray image (PNG/JPG)
2. **Model predicts** whether the scan shows Pneumonia or Normal findings
3. **Gemini AI generates** a detailed medical report with diagnosis explanation, precautions, and discharge instructions
4. **PDF report** is automatically created and emailed to the patient

---

## 🏗️ Architecture

```
X-ray Image
     │
     ▼
┌─────────────────────────────┐
│   ResNet50 (Feature Extractor)  │  ← Pretrained on ImageNet
│   CLIP Vision Model             │  ← Pretrained on OpenAI CLIP
└─────────────┬───────────────┘
              │ Combined Features
              ▼
     Fully Connected Layers
              │
              ▼
     Pneumonia / Normal
              │
              ▼
     Google Gemini AI
              │
              ▼
     PDF Report → Email
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Model | ResNet50 + CLIP (Hybrid) |
| Framework | PyTorch, Transformers |
| LLM Report Generation | Google Gemini 2.0 Flash |
| PDF Generation | ReportLab |
| Frontend / UI | Streamlit |
| Email Delivery | SMTP (Gmail) |
| Language | Python |

---

## 📁 Project Structure

```
CT_Scan_mlproject/
├── app.py                  # Main Streamlit application
├── new_c.ipynb             # Model training notebook
├── resnet_clip_model.pth   # Trained model weights
├── requirements.txt        # Python dependencies
├── trademark.png           # Watermark for PDF report
├── .gitignore
└── README.md
```

---

## 🔬 Model Details

- **Base Models:** ResNet50 (ImageNet pretrained) + CLIP ViT-Base-Patch32
- **Architecture:** Hybrid feature fusion — ResNet50 and CLIP visual features are concatenated and passed through fully connected layers
- **Classes:** Binary classification — `Pneumonia` vs `Normal`
- **Freezing Strategy:** ResNet backbone frozen except Layer4; CLIP vision encoder fully frozen
- **Output:** Class prediction + confidence score

---

## 📋 Features

- ✅ Real-time chest X-ray classification
- ✅ Confidence score for every prediction
- ✅ AI-generated medical report (Diagnosis, Precautions, Discharge Instructions)
- ✅ Auto-generated PDF with patient details and watermark
- ✅ Automated email delivery of report to patient
- ✅ Clean, interactive Streamlit UI
- ✅ Patient form — Name, Age, Sex, Email

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dhinakar07/CT_Scan_mlproject.git
cd CT_Scan_mlproject
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your API Keys
In `app.py`, replace the following with your actual keys:
```python
GEMINI_API_KEY = "your-gemini-api-key"
```

For email delivery, update:
```python
from_email = "your-email@gmail.com"
password = "your-app-password"
```

> ⚠️ Use a Gmail App Password (not your regular password). Enable 2FA and generate an App Password from your Google Account settings.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
torch
torchvision
transformers==4.44.2
streamlit
Pillow
numpy
google-generativeai
reportlab
```

---

## 📊 How to Use

1. Run the app with `streamlit run app.py`
2. Fill in patient details — Name, Age, Sex, Email
3. Upload a chest X-ray image (PNG, JPG, JPEG)
4. Click **Submit**
5. View the AI diagnosis and confidence score
6. Preview the generated medical report
7. Receive the full PDF report in your email inbox

---

## 📄 Sample Report Sections

The AI-generated medical report includes:

- **Diagnosis Explanation** — What the diagnosis means, causes, and clinical implications
- **Recommended Precautions or Next Steps** — Actionable steps for the patient
- **Discharge Instructions** — Follow-up care and monitoring guidance

---

## ⚠️ Disclaimer

This application is built for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a licensed medical professional for clinical decisions.

---

## 👨‍💻 Author

**Dhinakar Yalla**
Master's in Data Science — University at Buffalo, SUNY
📧 dhinakaryalla07@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/dhinakaryalla/)
🐙 [GitHub](https://github.com/dhinakar07)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
