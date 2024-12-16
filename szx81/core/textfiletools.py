from pathlib import Path
import re

from pdfminer.high_level import extract_text
import chardet

class TextData:
    def __init__(self, file_path, data_file_path=None):
        self.file_path = file_path
        self.data_file_path = self.file_path \
            if data_file_path is None else data_file_path
        self.encoding = None
        self.text = ''

    def pdf_to_text(self):
        self.text = extract_text(self.file_path)
        return self.text
    

    def detect_encoding(self):
        with open(self.file_path, 'rb') as file:
            detector = chardet.universaldetector.UniversalDetector()
            for line in file:
                detector.feed(line)
                if detector.done:
                    break
                detector.close()
            return detector.result['encoding']


    def read_file(self):
        self.text = ''
        encoding = self.detect_encoding()
        with open(self.file_path, 'r', encoding=encoding) as file:
            for line in file:
                self.text += line.strip()
        return self.text


    def pdf_to_text(self):
        try:
            self.text = self.pdf_to_text()
        except Exception as ex:
            if str(ex) == 'No /Root object! - Is this really a PDF?':
                return self.read_file()
            else:
                raise
        return self.text

    def write_file(self):
        self.data_file_path = self.data_file_path.replace('.pdf', '.txt')
        with open(
                self.data_file_path, 'w', 
                encoding=self.encoding) as file:
            file.write(self.text)

    def clean_learnenglish(self):
        """
        # https://learnenglish-new.com/short-stories-in-english-for-beginners-pdf/
        """
        text = self.pdf_to_text(self.file_path)
        text = re.sub(r'earnenglish-new.com', ' ', text)
        text = re.sub(r'\s+$', '', text)
        text = re.sub(r'\. *', '.\n', text)
        text = re.sub(r'-+ Page \d+-+', ' ', text)
        text = re.sub(r' +', ' ', text)
        self.text = text
        self.write_file()

class TokenIter:
    """Given a text data file, produce sequential words.
    """
    def __init__(self, file_path):
        self.tokens = []
        self.file_path = file_path
        self.line_iterator = None
        self.counter = None
    
    def __iter__(self):
        self.file = open(self.file_path, 'r', encoding = 'cp850')
        self.line_iterator = iter(self.file.readline, '')
        return self

    def __next__(self):
        if len(self.tokens) == 0:
            while len(self.tokens) == 0:
                line = next(self.line_iterator)
                if line == '':
                    raise StopIteration
                self.tokens = line.split()
        token = self.tokens.pop(0)
        return token
    
class LineIter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.line_iterator = None
        self.counter = None
    
    def __iter__(self):
        self.file = open(self.file_path, 'r', encoding = 'cp850')
        self.line_iterator = iter(self.file.readline, '')
        return self

    def __next__(self):
        return next(self.line_iterator)