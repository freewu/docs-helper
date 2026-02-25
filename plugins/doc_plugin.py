"""
DOC/DOCX document processing plugin
"""
from plugins.base import DocumentPlugin
from docx import Document
import win32com.client as win32
import os


class DocPlugin(DocumentPlugin):
    """Plugin for processing DOC/DOCX documents"""
    
    def get_supported_extensions(self) -> list:
        return ['.doc', '.docx']
    
    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.docx':
            return self._extract_docx(file_path)
        elif ext == '.doc':
            return self._extract_doc(file_path)
        else:
            return ""
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception:
            return ""
    
    def _extract_doc(self, file_path: str) -> str:
        """Extract text from DOC file using COM interface"""
        try:
            word_app = win32.Dispatch("Word.Application")
            word_app.Visible = False
            doc = word_app.Documents.Open(file_path)
            text = doc.Range().Text
            doc.Close()
            word_app.Quit()
            # Remove the extra characters Word adds
            return text.rstrip('\x07\x00')
        except Exception:
            # Fallback: try using docx converter if available
            try:
                import docx2txt
                return docx2txt.process(file_path)
            except ImportError:
                return ""
            except Exception:
                return ""