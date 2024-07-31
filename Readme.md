### README.md

# Invoice Similarity Detection

This project aims to identify the most similar invoice in a training set based on a new invoice. The similarity is determined using various metrics, including cosine similarity, Jaccard similarity, and structural similarity (SSIM) between images of the invoices.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Setup directories:**
   - Place your training invoices (PDF files) in a directory named `train`.
   - Place your test invoices (PDF files) in a directory named `test`.

### Running the Program

1. **Run the similarity detection script:**
   ```sh
   python <script-name>.py
   ```
   Replace `<script-name>` with the actual name of the script file containing the provided code.

2. **Output:**
   The program will output the most similar training invoice for each test invoice along with the similarity scores for cosine, Jaccard, and image similarities.

## Dependencies

- PyPDF2: To extract text from PDF files.
- NumPy: For numerical operations.
- Scikit-learn: For text feature extraction and similarity calculations.
- OpenCV: For image processing.
- pdf2image: To convert PDF pages to images.

### Approach Document -> Check out [Approach.md](https://github.com/Sam-wiz/Invoice_similarity_detector/blob/master/Approach.md)
