"""
EPUB document processing plugin
"""
from plugins.base import DocumentPlugin
from ebooklib import epub
import zipfile
from bs4 import BeautifulSoup


class EpubPlugin(DocumentPlugin):
    """Plugin for processing EPUB documents"""
    
    def get_supported_extensions(self) -> list:
        return ['.epub']
    
    def extract_text(self, file_path: str) -> str:
        try:
            # Method 1: Try using ebooklib
            book = epub.read_epub(file_path)
            text_parts = []
            
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text_parts.append(soup.get_text())
            
            return '\n'.join(text_parts)
        except Exception:
            # Fallback: Try extracting as a ZIP archive with HTML parsing
            try:
                text_parts = []
                with zipfile.ZipFile(file_path, 'r') as epub_zip:
                    # Get all HTML files in the EPUB
                    for file_info in epub_zip.filelist:
                        if file_info.filename.endswith('.xhtml') or file_info.filename.endswith('.html'):
                            with epub_zip.open(file_info) as html_file:
                                content = html_file.read()
                                soup = BeautifulSoup(content, 'html.parser')
                                text_parts.append(soup.get_text())
                return '\n'.join(text_parts)
            except Exception:
                return ""