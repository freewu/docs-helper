"""
Document processing plugins package
"""
from .base import PluginManager, DocumentPlugin
from .txt_plugin import TxtPlugin
from .doc_plugin import DocPlugin 
from .pdf_plugin import PdfPlugin
from .epub_plugin import EpubPlugin


def get_default_plugins():
    """Get all default document processing plugins"""
    return [
        TxtPlugin(),
        DocPlugin(), 
        PdfPlugin(),
        EpubPlugin()
    ]


def get_plugin_manager():
    """Initialize and return a plugin manager with default plugins"""
    manager = PluginManager()
    for plugin in get_default_plugins():
        manager.register_plugin(plugin)
    return manager