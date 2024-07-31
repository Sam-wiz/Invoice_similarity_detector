import os
import re
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from skimage.metrics import structural_similarity as ssim
import cv2
from pdf2image import convert_from_path

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def extract_features(text):
    words = re.findall(r'\w+', text.lower())
    dates = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
    amounts = re.findall(r'\b\d+\.\d{2}\b', text)
    invoice_number = re.search(r'invoice\s*number\s*[:#]*\s*(\w+)', text, re.IGNORECASE)
    
    return {
        'words': words,
        'dates': dates,
        'amounts': amounts,
        'invoice_number': invoice_number.group(1) if invoice_number else None,
        'full_text': ' '.join(words + dates + amounts)
    }

def calculate_cosine_similarity(new_invoice_features, train_invoices_features):
    vectorizer = TfidfVectorizer()
    database_texts = [features['full_text'] for features in train_invoices_features]
    vectors = vectorizer.fit_transform([new_invoice_features['full_text']] + database_texts)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:])
    return cosine_sim[0]

def calculate_jaccard_similarity(new_invoice_features, train_invoices_features):
    vectorizer = CountVectorizer(binary=True)
    database_texts = [' '.join(features['words']) for features in train_invoices_features]
    vectors = vectorizer.fit_transform([new_invoice_features['full_text']] + database_texts).toarray()
    intersection = np.minimum(vectors[0], vectors[1:]).sum(axis=1)
    union = np.maximum(vectors[0], vectors[1:]).sum(axis=1)
    jaccard_sim = intersection / union
    return jaccard_sim

def convert_pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if images:
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    return None

def calculate_image_similarity(new_invoice_path, train_invoice_path):
    new_image = convert_pdf_to_image(new_invoice_path)
    train_image = convert_pdf_to_image(train_invoice_path)
    if new_image is None or train_image is None:
        return 0 
    
    height = min(new_image.shape[0], train_image.shape[0])
    width = min(new_image.shape[1], train_image.shape[1])
    new_image_resized = cv2.resize(new_image, (width, height))
    train_image_resized = cv2.resize(train_image, (width, height))
    
    new_gray = cv2.cvtColor(new_image_resized, cv2.COLOR_BGR2GRAY)
    train_gray = cv2.cvtColor(train_image_resized, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(new_gray, train_gray, full=True)
    return score

def load_invoices_from_directory(directory_path):
    invoices = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(file_path)
            features = extract_features(text)
            invoices.append({'path': file_path, 'features': features})
    return invoices

def find_most_similar_invoice(new_invoice_features, new_invoice_path, train_invoices):
    train_invoices_features = [invoice['features'] for invoice in train_invoices]
    
    cosine_similarities = calculate_cosine_similarity(new_invoice_features, train_invoices_features)
    jaccard_similarities = calculate_jaccard_similarity(new_invoice_features, train_invoices_features)
    image_similarities = [calculate_image_similarity(new_invoice_path, invoice['path']) for invoice in train_invoices]
    
    most_similar_index = cosine_similarities.argmax()
    
    most_similar_invoice = train_invoices[most_similar_index]
    similarity_scores = {
        'cosine': cosine_similarities[most_similar_index],
        'jaccard': jaccard_similarities[most_similar_index],
        'image': image_similarities[most_similar_index]
    }
    average_similarity = np.mean(list(similarity_scores.values())) # I have given equal weightage to all the three methods
    return most_similar_invoice, similarity_scores, average_similarity

def process_test_invoices(train_directory, test_directory):
    train_invoices = load_invoices_from_directory(train_directory)
    
    for filename in os.listdir(test_directory):
        if filename.endswith('.pdf'):
            test_file_path = os.path.join(test_directory, filename)
            test_text = extract_text_from_pdf(test_file_path)
            test_features = extract_features(test_text)
            
            most_similar_invoice, similarity_scores, average_similarity = find_most_similar_invoice(
                test_features, test_file_path, train_invoices)
            
            print(f"Test Invoice: {filename}")
            print(f"Most Similar Invoice: {most_similar_invoice['path']}")
            print(f"  Cosine Similarity: {similarity_scores['cosine']:.2f}")
            print(f"  Jaccard Similarity: {similarity_scores['jaccard']:.2f}")
            print(f"  Image Similarity: {similarity_scores['image']:.2f}")
            print(f"  Average Similarity: {average_similarity:.2f}")
            print()

train_directory = 'train'  
test_directory = 'test'    
process_test_invoices(train_directory, test_directory)
