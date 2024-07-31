import os
import pdfplumber
import cv2
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Function to extract text and layout from PDF using pdfplumber
def extract_text_with_layout(file_path):
    text_with_bbox = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for word in page.extract_words():
                text_with_bbox.append((word['text'], word['x0'], word['top'], word['x1'], word['bottom'], page.page_number))
    return text_with_bbox

# Function to extract text from PDF using PyPDF2 (for comparison purposes)
def extract_text(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Load and extract text from training and test PDFs
train_folder = 'train'
test_folder = 'test'

train_files = [file for file in os.listdir(train_folder) if file.endswith('.pdf')]
test_files = [file for file in os.listdir(test_folder) if file.endswith('.pdf')]

train_texts = [extract_text(os.path.join(train_folder, file)) for file in train_files]
test_texts = [extract_text(os.path.join(test_folder, file)) for file in test_files]

# Extract text with layout
train_texts_with_layout = [extract_text_with_layout(os.path.join(train_folder, file)) for file in train_files]
test_texts_with_layout = [extract_text_with_layout(os.path.join(test_folder, file)) for file in test_files]

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Calculate cosine similarity
def find_most_similar(train_vectors, test_vector):
    similarities = cosine_similarity(train_vectors, test_vector.reshape(1, -1))
    similarities = similarities.flatten()
    most_similar_index = similarities.argmax()
    return most_similar_index, similarities[most_similar_index]

# Highlight similar text
def highlight_similar_text(train_layout, test_layout, train_pdf_path, test_pdf_path, output_path):
    # Convert PDF pages to images
    test_pages = convert_from_path(test_pdf_path)

    # Convert images to OpenCV format
    test_images = [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in test_pages]

    # Highlight similar text in test document
    for word in test_layout:
        if word[0] in [t[0] for t in train_layout]:
            page_num = word[5] - 1
            x0, y0, x1, y1 = int(word[1]), int(word[2]), int(word[3]), int(word[4])
            cv2.rectangle(test_images[page_num], (x0, y0), (x1, y1), (0, 255, 0), 2)  # Use green color for better visibility

    # Save highlighted images
    for i, img in enumerate(test_images):
        output_img_path = f"{output_path}_page_{i + 1}.png"
        cv2.imwrite(output_img_path, img)

# Process each test document
for test_index, test_vector in enumerate(test_vectors):
    most_similar_index, similarity_score = find_most_similar(train_vectors, test_vector)
    similarity_score = similarity_score.item()
    print(f"Test Document {test_index+1} is most similar to Train Document {most_similar_index+1} with a similarity score of {similarity_score:.2f}")

    # Highlight similar text
    highlight_similar_text(
        train_texts_with_layout[most_similar_index],
        test_texts_with_layout[test_index],
        os.path.join(train_folder, train_files[most_similar_index]),
        os.path.join(test_folder, test_files[test_index]),
        f"output_test_{test_index + 1}_train_{most_similar_index + 1}"
    )
