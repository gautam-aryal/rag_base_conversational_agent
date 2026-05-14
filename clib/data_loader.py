import pymupdf4llm
import PyPDF2
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
# Load raw data form PDF, text file etc
class DataLoader:

  def __init__(self, path):
    self.path =  path

  def load_raw_text(self):
    pass

  def load_markup_text(self):
    text = pymupdf4llm.to_markdown(
        self.path,
        page_chunks = False

    )
    return text
  
  def file_name_sort(self, files):
    names = [f for f in files]
    names.sort()
    return names


  def extract_text_from_multiple_pdf(self, data_dir: str = 'data'):
    # Load and process PDF documents
    documents = []
    for filename in self.file_name_sort(os.listdir(data_dir)):
        if filename.endswith(".pdf"):
            print("Reading from: " + filename)
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            documents.extend(loader.load())
    return documents

  def extract_text_from_single_pdf(self):
    text = ""
    with open(self.path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
            # Clean up whitespace
            text = text.strip()
    return text


  def extract_markup_text_from_single_pdf(self):
    text = pymupdf4llm.to_markdown(
        self.path,
        page_chunks = False

    )
    return text

  def extract_markup_text_from_multiple_pdf(self):
    text = ""
    for filename in self.file_name_sort(os.listdir(self.path)):
        if filename.endswith(".pdf"):
          print("Reading from: " + filename)
          text += pymupdf4llm.to_markdown(
                os.path.join(self.path, filename),
                page_chunks = False
          )
    return text
