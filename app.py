import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import logging
import re
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError as e:
    st.error(f"Failed to import CLIPProcessor/CLIPModel: {e}. Please install transformers==4.44.2")
    raise e
from torchvision import models
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyD2o8aUh1HXjVL1U_wm4NoR5WBpy2U_knw"  # Replace with your actual Gemini API key
if not GEMINI_API_KEY:
    st.error("Google Gemini API key is missing.")
    raise ValueError("GEMINI_API_KEY is required")
genai.configure(api_key=GEMINI_API_KEY)

# Define model class
class ResNetCLIPClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_resnet=True, freeze_clip=True):
        super(ResNetCLIPClassifier, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_features = self.clip_model.vision_model.config.hidden_size
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(resnet_features + clip_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, pixel_values):
        resnet_out = self.resnet(pixel_values)
        clip_out = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        combined = torch.cat((resnet_out, clip_out), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model
@st.cache_resource
def load_model():
    logger.info("Loading model...")
    device = torch.device('cpu')  # Use CPU for 16GB RAM
    try:
        model = ResNetCLIPClassifier(freeze_resnet=True, freeze_clip=True).to(device)
        model.load_state_dict(torch.load("resnet_clip_model.pth", map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise e
    logger.info("Model loaded successfully")
    return model, model.clip_processor, device

# Predict pneumonia
def predict_pneumonia(model, processor, device, image):
    logger.info("Predicting pneumonia...")
    img = image.convert('RGB')
    inputs = processor(images=img, return_tensors="pt", padding=True)
    pixel_values = inputs['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    label = "Pneumonia" if pred_class == 1 else "Normal"
    logger.info(f"Prediction: {label}, Confidence: {confidence:.4f}")
    return label, confidence

# Generate LLM report using Google Gemini
def generate_llm_report(name, age, sex, email, diagnosis, confidence):
    logger.info("Generating LLM report with Google Gemini...")
    prompt = f"""
    You are a medical expert. A patient has undergone an X-ray scan, diagnosed as {diagnosis} with {confidence*100:.1f}% confidence.
    Generate a detailed medical report with:
    - Diagnosis Explanation: Describe pneumonia or normal findings, causes, and implications.
    - Recommended Precautions or Next Steps: Use bullet points.
    - Discharge Instructions: Use bullet points.
    Do NOT include patient details (name, age, sex, email) or a 'Medical Report' title.
    Use a professional tone, limit to 500 words, and structure with sections titled 'Diagnosis Explanation', 'Recommended Precautions or Next Steps', 'Discharge Instructions'.
    Avoid markdown bold (**text**) or other formatting symbols.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        report = response.text
    except Exception as e:
        st.error(f"Failed to generate LLM report with Gemini: {e}")
        raise e
    logger.info("LLM report generated")
    return report

# Create PDF with proper formatting
def create_pdf_report(name, age, sex, email, diagnosis, confidence, llm_report, output_path):
    logger.info("Creating PDF report...")
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=20,
        alignment=1
    )
    section_style = ParagraphStyle(
        name='Section',
        parent=styles['Heading2'],
        fontSize=12,
        fontName='Helvetica-Bold',
        spaceBefore=15,
        spaceAfter=10
    )
    normal_style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        spaceAfter=8
    )
    bullet_style = ParagraphStyle(
        name='Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=5
    )
    footer_style = ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        spaceBefore=15
    )

    def add_background(canvas, doc):
        page_width, page_height = letter
        canvas.saveState()
        canvas.setFillAlpha(0.2)
        canvas.drawImage('trademark.png', 
                         (page_width - 100) / 2,
                         (page_height - 100) / 2,
                         width=100, height=100,
                         mask='auto')
        canvas.restoreState()

    content = []
    content.append(Paragraph("Pneumonia Detection Report", title_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Name:</b> {name}", normal_style))
    content.append(Paragraph(f"<b>Age:</b> {age}", normal_style))
    content.append(Paragraph(f"<b>Sex:</b> {sex}", normal_style))
    content.append(Paragraph(f"<b>Email:</b> {email}", normal_style))
    content.append(Paragraph(f"<b>Diagnosis:</b> {diagnosis} ({confidence*100:.1f}% confidence)", normal_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Medical Report", section_style))
    
    llm_report = re.sub(r'\*\*(.*?)\*\*', r'\1', llm_report)
    lines = llm_report.replace('\n', '<br/>').split('<br/>')
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(('diagnosis explanation', 'recommended precautions or next steps', 'discharge instructions')):
            current_section = line
            content.append(Paragraph(line, section_style))
            content.append(Spacer(1, 8))
        elif line.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ')):
            bullet_text = line[2:] if line.startswith(('- ', '* ', '• ')) else line[3:]
            bullet_text = bullet_text.strip()
            content.append(Paragraph(f"• {bullet_text}", bullet_style))
            content.append(Spacer(1, 3))
        else:
            if line.lower().startswith('please note'):
                content.append(Paragraph(line, footer_style))
                content.append(Spacer(1, 10))
            else:
                content.append(Paragraph(line, normal_style))
                content.append(Spacer(1, 5))
    
    try:
        doc.build(content, onFirstPage=add_background, onLaterPages=add_background)
        logger.info("PDF created successfully")
    except Exception as e:
        st.error(f"Failed to create PDF: {e}")
        raise e

# Send email
def send_email(to_email, pdf_path, name):
    logger.info("Sending email...")
    from_email = "dhinakaryalla@gmail.com"
    password = "phmf tqkx pmpe amul"
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Your Pneumonia Detection Report"
    body = f"Dear {name},\n\nAttached is your pneumonia detection report.\n\nBest regards,\nAI Health Team"
    msg.attach(MIMEText(body, 'plain'))
    try:
        with open(pdf_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=report.pdf")
        msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        logger.info("Email sent successfully")
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        raise e

# Streamlit UI
st.title("Pneumonia Detection from X-ray")
st.write("Upload an X-ray image and provide your details to receive a diagnosis and report.")

with st.form("user_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    email = st.text_input("Email")
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Submit")

if submitted and uploaded_file is not None:
    try:
        model, processor, device = load_model()
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", width=300)
        with st.spinner("Analyzing X-ray..."):
            diagnosis, confidence = predict_pneumonia(model, processor, device, image)
        st.success(f"Diagnosis: {diagnosis} ({confidence*100:.1f}% confidence)")
        with st.spinner("Generating medical report..."):
            llm_report = generate_llm_report(name, age, sex, email, diagnosis, confidence)
        st.write("Medical Report Preview:")
        st.write(llm_report)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            create_pdf_report(name, age, sex, email, diagnosis, confidence, llm_report, pdf_path)
        with st.spinner("Sending report to email..."):
            send_email(email, pdf_path, name)
        st.success(f"Report sent to {email}")
        os.remove(pdf_path)
    except Exception as e:
        st.error(f"Error: {str(e)}")