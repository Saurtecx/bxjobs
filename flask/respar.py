import PyPDF2
import docx
import os
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Define a list of skill keywords or phrases
skill_keywords = ['Python', 'Java', 'Machine Learning', 'Data Analysis', 'C++', 'MySQL' , 'Algorithms']

def extract_resume_text(filename):
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.pdf':
        with open(filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            resume_text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                resume_text += page.extract_text()
    elif ext == '.docx':
        doc = docx.Document(filename)
        resume_text = ''
        for para in doc.paragraphs:
            resume_text += para.text
    elif ext == '.txt':
        with open(filename, 'r') as file:
            resume_text = file.read()
    else:
        raise ValueError(f'Unsupported file format: {ext}')
    
    # Preprocess the resume text
    sentences = sent_tokenize(resume_text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    skills = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        for keyword in skill_keywords:
            if keyword.lower() in stemmed_words:
                skills.append(keyword)
    
    return skills

# Test the function with different file types
pdf_skills = extract_resume_text('resume.pdf')
print('PDF Skills:', pdf_skills)

# docx_skills = extract_resume_text('resume.docx')
# print('DOCX Skills:', docx_skills)

# txt_skills = extract_resume_text('resume.txt')
# print('TXT Skills:', txt_skills)
