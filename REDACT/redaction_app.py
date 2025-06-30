import streamlit as st
import spacy
from spacy import displacy
import io
from cryptography.fernet import Fernet
from flair.models import TextClassifier
from flair.data import Sentence
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Load necessary models
nlp = spacy.load('en_core_web_lg')  # For English NER, POS, dependency parsing, etc.
sentiment_analyzer = SentimentIntensityAnalyzer()  # Sentiment analysis using NLTK

# Function to encrypt the data
def encrypt_data(data):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data, cipher_suite

# Function to decrypt the data
def decrypt_data(encrypted_data, cipher_suite):
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data

# Function to perform text classification using Flair
def classify_text(text):
    classifier = TextClassifier.load('en-sentiment')
    sentence = Sentence(text)
    classifier.predict(sentence)
    
    # Handle case where no labels are returned
    if sentence.labels:
        return sentence.labels[0]
    else:
        return None

# Function to generate synthetic text using spaCy
def generate_synthetic_text(redacted_text):
    return redacted_text.replace("[REDACTED]", "SYNTHETIC_DATA")

# Function to map entity types to synthetic placeholders
def entity_placeholder(entity_label):
    placeholder_map = {
        "PERSON": "[NAME]",
        "ORG": "[ORGANIZATION]",
        "GPE": "[LOCATION]",
        "LOC": "[LOCATION]",
        "DATE": "[DATE]",
        "MONEY": "[AMOUNT]",
        "CARDINAL": "[NUMBER]"
    }
    return placeholder_map.get(entity_label, "[REDACTED]")  # Default to [REDACTED] if not in map

# Main redaction function integrating all features
def perform_advanced_redaction(text, redaction_level):
    doc = nlp(text)
    entities_to_redact = []

    # Define redaction levels
    level_map = {
        1: ["PERSON"],
        2: ["PERSON", "ORG"],
        3: ["PERSON", "ORG", "GPE", "LOC"],
        4: ["PERSON", "ORG", "GPE", "LOC", "DATE"],
        5: ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "CARDINAL"]
    }

    if redaction_level in level_map:
        entities_to_redact = level_map[redaction_level]

    redacted_text = text
    for ent in doc.ents:
        if ent.label_ in entities_to_redact:
            placeholder = entity_placeholder(ent.label_)
            redacted_text = redacted_text.replace(ent.text, placeholder)

    # Sentiment analysis to identify emotionally charged text
    sentiment_score = sentiment_analyzer.polarity_scores(redacted_text)
    if sentiment_score['compound'] < -0.3:  # Threshold for negative sentiment
        redacted_text = "[REDACTED DUE TO SENSITIVE SENTIMENT]"

    # Text classification using Flair
    classification = classify_text(redacted_text)
    if classification and "negative" in classification.value:
        redacted_text = "[REDACTED DUE TO NEGATIVE CLASSIFICATION]"

    # Generate synthetic text for redacted content
    synthetic_text = generate_synthetic_text(redacted_text)

    return synthetic_text, displacy.render(doc, style="ent")

# Streamlit UI
def main():
    st.title("Advanced Document Redaction Application")

    # Upload document
    uploaded_file = st.file_uploader("Upload a document (TXT format)", type=["txt"])

    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            # Try a different encoding if UTF-8 fails
            file_contents = uploaded_file.read().decode("ISO-8859-1")

        # Encrypt the uploaded file contents
        encrypted_data, cipher_suite = encrypt_data(file_contents)
        st.success("File uploaded and encrypted successfully!")

        # Decrypt and preview file
        if st.checkbox("Preview Original Document"):
            decrypted_data = decrypt_data(encrypted_data, cipher_suite)
            st.text_area("Original Document", value=decrypted_data, height=300)

        # Redaction Level
        st.subheader("Set Redaction Level")
        redaction_level = st.slider("Select a redaction level (1-5):", 1, 5, 3)

        # Perform Redaction with Advanced NLP
        if st.button("Redact Document"):
            decrypted_data = decrypt_data(encrypted_data, cipher_suite)
            redacted_text, annotated_html = perform_advanced_redaction(decrypted_data, redaction_level)
            st.subheader("Redacted Document Preview")
            st.text_area("Redacted Document", value=redacted_text, height=300)

            # Display named entity annotations
            st.subheader("NER Annotations")
            st.write(annotated_html, unsafe_allow_html=True)

            # Download redacted document
            st.download_button(label="Download Redacted Document", data=redacted_text, file_name="redacted_document.txt", mime="text/plain")


# Run the app
if __name__ == '__main__':
    main()
