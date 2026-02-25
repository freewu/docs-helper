"""
TXT document processing plugin
"""
from plugins.base import DocumentPlugin


class TxtPlugin(DocumentPlugin):
    """Plugin for processing TXT documents"""
    
    def get_supported_extensions(self) -> list:
        return ['.txt']
    
    def extract_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            try:
                with open(file_path, 'r', encoding='gbk') as file:
                    return file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
        except Exception:
            return ""