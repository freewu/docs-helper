"""
PDF document processing plugin
"""
from plugins.base import DocumentPlugin
import PyPDF2


class PdfPlugin(DocumentPlugin):
    """Plugin for processing PDF documents"""
    
    def get_supported_extensions(self) -> list:
        return ['.pdf']
    
    def extract_text(self, file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception:
            return ""