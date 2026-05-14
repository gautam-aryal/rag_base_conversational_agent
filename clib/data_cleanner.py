import re
from langchain_core.documents import Document

class DataCleanner:
  def __init__(self, text):
    pass

  @staticmethod
  def remove_html(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

  @staticmethod
  def remove_other(text):
    clean_text = re.sub(r'\.{3,}', ' ', text)
    return clean_text


  @staticmethod
  def remove_markdown(text):
    # Remove headings (#, ##,...)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold (**bold** or __bold__)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    return text

  @staticmethod
  def remove_extra_space(text):
    # Step 1: Replace one or more whitespace characters (space, tab, newline, etc.) with a single space
    cleaned_text = re.sub(r'\s+', ' ', text)

    # Step 2: Remove leading and trailing white spaces from the final string
    cleaned_text = cleaned_text.strip()

    return cleaned_text

  @staticmethod
  def clean_chain_all(text):
    text = DataCleanner.remove_html(text)
    text = DataCleanner.remove_markdown(text)
    text = DataCleanner.remove_extra_space(text)
    text = DataCleanner.remove_other(text)
    return text

  @staticmethod
  def clean_docs(docs):
    cleaned_docs = []
    for doc in docs:
      cleaned_text = DataCleanner.clean_chain_all(doc.page_content)

      cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
    return cleaned_docs

  @staticmethod
  def clean_text(text):
    return Data.clean_chain_all(text)