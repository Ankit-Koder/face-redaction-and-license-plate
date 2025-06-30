import streamlit as st
import spacy
from spacy import displacy
import io
from cryptography.fernet import Fernet
from faker import Faker
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
import docx
import openpyxl
from PyPDF2 import PdfReader, PdfWriter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load necessary models
nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_lg')  # SpaCy NER
sentiment_analyzer = SentimentIntensityAnalyzer()  # NLTK Sentiment Analysis
classifier = TextClassifier.load('en-sentiment')  # Flair Sentiment Analysis
fake = Faker()  # For generating fake synthetic data

# Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Function to encrypt data
def encrypt_data(data):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data, cipher_suite

# Function to decrypt data
def decrypt_data(encrypted_data, cipher_suite):
    return cipher_suite.decrypt(encrypted_data).decode()

# Function to classify text using Flair
def classify_text(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    if sentence.labels:
        return sentence.labels[0]
    return None

# Function to process sentiment
def perform_sentiment_redaction(text):
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    if sentiment_score['compound'] < -0.3:  # Threshold for negative sentiment
        return "[REDACTED DUE TO SENSITIVE SENTIMENT]"
    
    classification = classify_text(text)
    if classification and "negative" in classification.value:
        return "[REDACTED DUE TO NEGATIVE CLASSIFICATION]"
    
    return text

# Function to generate synthetic text replacements
def synthetic_data_replacement(entity_label):
    fake_data_map = {
        "PERSON": fake.name(),
        "ORG": fake.company(),
        "GPE": fake.city(),
        "LOC": fake.address(),
        "DATE": fake.date(),
        "MONEY": fake.pricetag(),
        "CARDINAL": fake.random_number(),
        "ID": fake.bothify(text="???######").upper(),
    }
    return fake_data_map.get(entity_label, "[REDACTED]")

# Function to handle text redaction
def redact_text(text, redaction_level, use_fake_data=False):
    doc = nlp(text)
    entities_to_redact = []
    
    # Define redaction levels with associated entities
    level_map = {
        1: ["PERSON"],
        2: ["PERSON", "ORG"],
        3: ["PERSON", "ORG", "GPE", "LOC"],
        4: ["PERSON", "ORG", "GPE", "LOC", "DATE"],
        5: ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "CARDINAL"],
    }

    if redaction_level in level_map:
        entities_to_redact = level_map[redaction_level]
    
    redacted_text = text

    # Loop through entities and replace or redact
    for ent in doc.ents:
        if ent.label_ in entities_to_redact:
            if use_fake_data:
                # Replace with synthetic data
                replacement = synthetic_data_replacement(ent.label_)
            else:
                # Replace with '[REDACTED]' tag
                replacement = "[REDACTED]"
            
            redacted_text = redacted_text.replace(ent.text, replacement)

    # Handle sentiment redaction
    redacted_text = perform_sentiment_redaction(redacted_text)
    
    # Use Presidio for NER-based redaction
    analyzer_results = analyzer.analyze(text, language="en")
    anonymized_results = anonymizer.anonymize(text=redacted_text, analyzer_results=analyzer_results)
    
    return anonymized_results.text, displacy.render(doc, style="ent")

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1]
    file_contents = ""

    if file_extension == 'txt':
        file_contents = uploaded_file.read().decode("utf-8")
    elif file_extension == 'docx':
        doc = docx.Document(uploaded_file)
        file_contents = "\n".join([p.text for p in doc.paragraphs])
    elif file_extension == 'pdf':
        reader = PdfReader(uploaded_file)
        file_contents = ""
        for page in reader.pages:
            file_contents += page.extract_text()
    elif file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)
        file_contents = df.to_string()
    else:
        st.error("Unsupported file format")
        return None, file_extension

    return file_contents, file_extension

# Function to convert text to PDF
def convert_to_pdf(text, output_filename):
    pdf_writer = PdfWriter()
    pdf_page = pdf_writer.add_blank_page(width=200, height=300)
    pdf_writer.add_page(pdf_page)
    
    with open(output_filename, 'wb') as pdf_output_file:
        pdf_writer.write(pdf_output_file)

# Streamlit UI
def main():
    st.title("Advanced Redaction Application")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a document (.txt, .docx, .pdf, .xlsx)", type=["txt", "docx", "pdf", "xlsx"])
    
    if uploaded_file is not None:
        try:
            file_contents, file_extension = extract_text_from_file(uploaded_file)

            if file_contents is None:
                return

            # Encrypt the file
            encrypted_data, cipher_suite = encrypt_data(file_contents)
            st.success("File uploaded and encrypted successfully!")
            
            # Preview original document
            if st.checkbox("Preview Original Document"):
                decrypted_data = decrypt_data(encrypted_data, cipher_suite)
                st.text_area("Original Document", value=decrypted_data, height=300)
            
            # Redaction Level
            st.subheader("Set Redaction Level")
            redaction_level = st.slider("Select a redaction level (1-5):", 1, 5, 3)
            
            # Checkbox to replace redacted data with fake synthetic data
            use_fake_data = st.checkbox("Replace redacted data with fake synthetic data?")
            
            # Perform Redaction
            if st.button("Redact Document"):
                decrypted_data = decrypt_data(encrypted_data, cipher_suite)
                redacted_text, annotated_html = redact_text(decrypted_data, redaction_level, use_fake_data)
                
                # Redacted document preview
                st.subheader("Redacted Document Preview")
                st.text_area("Redacted Document", value=redacted_text, height=300)
                
                # Display NER Annotations
                st.subheader("NER Annotations")
                st.write(annotated_html, unsafe_allow_html=True)
                
                # Download redacted document
                download_format = st.selectbox("Download format", ["Original", "PDF"])
                if download_format == "PDF":
                    output_filename = f"redacted_document.pdf"
                    convert_to_pdf(redacted_text, output_filename)
                    with open(output_filename, 'rb') as file:
                        st.download_button(label="Download Redacted Document as PDF", data=file, file_name=output_filename, mime="application/pdf")
                else:
                    output_filename = f"redacted_document.{file_extension}"
                    st.download_button(label="Download Redacted Document", data=redacted_text, file_name=output_filename, mime="text/plain")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == '__main__':
    main()
