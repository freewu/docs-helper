"""
Base interface for document processing plugins
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class DocumentPlugin(ABC):
    """Base class for document processing plugins"""
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g., ['.txt', '.pdf'])"""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text content from the document"""
        pass


class PluginManager:
    """Manages document processing plugins"""
    
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, plugin: DocumentPlugin):
        """Register a document processing plugin"""
        for ext in plugin.get_supported_extensions():
            self.plugins[ext.lower()] = plugin
    
    def get_plugin_for_extension(self, extension: str) -> Optional[DocumentPlugin]:
        """Get the appropriate plugin for a file extension"""
        return self.plugins.get(extension.lower())
    
    def is_supported(self, extension: str) -> bool:
        """Check if the extension is supported"""
        return extension.lower() in self.plugins
    
    def get_all_supported_extensions(self) -> List[str]:
        """Get all supported extensions"""
        return list(self.plugins.keys())