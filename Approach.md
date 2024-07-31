# Invoice Similarity Detection

## Overview

This document outlines the approach and methodology used in the Invoice Similarity Detection project. The main goal of this project is to identify the most similar invoice in a training dataset based on a new invoice. The similarity is evaluated using three main metrics: cosine similarity, Jaccard similarity, and structural similarity (SSIM) between images.

## Approach

### 1. **Data Extraction**

#### a. **Text Extraction from PDFs**
   - **Function:** `extract_text_from_pdf(file_path)`
   - **Description:** This function extracts the textual content from a PDF file. It uses the PyPDF2 library to read each page and accumulate the text.

### 2. **Feature Extraction**

#### a. **Textual Features**
   - **Function:** `extract_features(text)`
   - **Description:** Extracts relevant features from the extracted text, including:
     - **Words:** All words in the text, converted to lowercase.
     - **Dates:** Dates in the format `DD/MM/YYYY`.
     - **Amounts:** Numerical values representing monetary amounts, captured in the format `xx.xx`.
     - **Invoice Number:** Extracts the invoice number using a regular expression pattern.

#### b. **Image Conversion**
   - **Function:** `convert_pdf_to_image(pdf_path)`
   - **Description:** Converts the first page of a PDF into an image using the `pdf2image` library, which can be used for image-based similarity comparison.

### 3. **Similarity Calculation**

#### a. **Cosine Similarity**
   - **Function:** `calculate_cosine_similarity(new_invoice_features, train_invoices_features)`
   - **Description:** Calculates the cosine similarity between the new invoice's text and the text of training invoices. The TfidfVectorizer from scikit-learn is used to vectorize the text data.

#### b. **Jaccard Similarity**
   - **Function:** `calculate_jaccard_similarity(new_invoice_features, train_invoices_features)`
   - **Description:** Calculates the Jaccard similarity based on the presence of words in the text. It uses the CountVectorizer to create binary vectors indicating word presence.

#### c. **Image Similarity**
   - **Function:** `calculate_image_similarity(new_invoice_path, train_invoice_path)`
   - **Description:** Computes the structural similarity index (SSIM) between images of two invoices. It first converts the PDF pages to images and then resizes them to a common size for comparison.

### 4. **Finding the Most Similar Invoice**

#### a. **Processing Invoices**
   - **Function:** `load_invoices_from_directory(directory_path)`
   - **Description:** Loads invoices from a specified directory, extracts their text, and computes the necessary features.

#### b. **Similarity Matching**
   - **Function:** `find_most_similar_invoice(new_invoice_features, new_invoice_path, train_invoices)`
   - **Description:** Determines the most similar invoice from the training set to a given test invoice. It uses a combination of cosine similarity, Jaccard similarity, and image similarity. The function returns the most similar invoice along with the similarity scores and an average similarity score.

### 5. **Execution Workflow**

#### a. **Main Execution Function**
   - **Function:** `process_test_invoices(train_directory, test_directory)`
   - **Description:** This is the main function that processes all test invoices. It loads training invoices, extracts features for each test invoice, calculates similarity scores, and identifies the most similar training invoice for each test invoice.

## Conclusion

The combination of textual and visual features ensures a comprehensive similarity comparison. The approach can be fine-tuned or extended by adjusting the weightage of different similarity metrics or including additional features. This document provides a foundational understanding of the methods and their roles in the Invoice Similarity Detection project.
