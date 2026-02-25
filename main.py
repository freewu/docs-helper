import sys
import os
import threading
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QTextEdit, QLabel, QLineEdit, QFileDialog, 
                              QMessageBox, QSpinBox, QListWidget, QListWidgetItem, QTextBrowser)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon, QTextCursor
import PyPDF2
from docx import Document
import pickle
from version import __version__
from plugins import get_plugin_manager

# Fix for PyInstaller compatibility with transformers library
if getattr(sys, 'frozen', False):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocumentProcessor(QObject):
    """Handles document processing and vector database operations"""
    
    # Signals for communication with GUI
    progress_updated = Signal(str)
    scanning_finished = Signal()
    
    def __init__(self):
        super().__init__()
        # Initialize the model with proper environment settings for PyInstaller
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Lightweight model for embeddings
        self.documents_data = []  # Store document content and metadata
        self.plugin_manager = get_plugin_manager()  # Initialize plugin manager
        
    def extract_text_from_file(self, file_path):
        """Extract text from various file types using plugins"""
        try:
            ext = Path(file_path).suffix.lower()
            
            # Check if the extension is supported by a plugin
            if self.plugin_manager.is_supported(ext):
                plugin = self.plugin_manager.get_plugin_for_extension(ext)
                if plugin:
                    return plugin.extract_text(file_path)
            return ""
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def scan_directory(self, directory_path):
        """Scan directory for supported documents and extract content"""
        self.progress_updated.emit("开始扫描目录...")
        
        # Supported extensions
        supported_ext = ['.txt', '.pdf', '.doc', '.docx']
        
        # Find all supported files
        files = []
        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_ext:
                    files.append(os.path.join(root, filename))
        
        total_files = len(files)
        self.progress_updated.emit(f"找到 {total_files} 个待处理文件...")
        
        # Process each file
        self.documents_data = []
        for i, file_path in enumerate(files):
            self.progress_updated.emit(f"正在处理文件 {i+1}/{total_files}: {os.path.basename(file_path)}")
            
            text_content = self.extract_text_from_file(file_path)
            if text_content.strip():  # Only store non-empty content
                # Split content into chunks to handle large documents
                chunks = self.split_text_into_chunks(text_content)
                
                for j, chunk in enumerate(chunks):
                    self.documents_data.append({
                        'file_path': file_path,
                        'chunk_index': j,
                        'content': chunk,
                        'original_filename': os.path.basename(file_path)
                    })
        
        self.progress_updated.emit(f"已处理来自 {total_files} 个文件的 {len(self.documents_data)} 个内容块。")
        self.scanning_finished.emit()
    
    def split_text_into_chunks(self, text, max_chunk_size=512):
        """Split text into smaller chunks to handle large documents"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def save_to_vector_db(self, db_path):
        """Save processed documents to vector database"""
        if not self.documents_data:
            return False
            
        # Create embeddings for all document chunks
        texts = [doc['content'] for doc in self.documents_data]
        embeddings = self.model.encode(texts)
        
        # Convert embeddings to numpy array (FAISS requirement)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index and documents data
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        faiss.write_index(index, db_path)
        
        # Save document metadata separately
        metadata_path = db_path.replace('.index', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents_data, f)
        
        return True
    
    def load_from_vector_db(self, db_path):
        """Load vector database and associated metadata"""
        try:
            index = faiss.read_index(db_path)
            
            # Load metadata
            metadata_path = db_path.replace('.index', '_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                self.documents_data = pickle.load(f)
            
            return index
        except Exception as e:
            print(f"Error loading vector database: {str(e)}")
            return None
    
    def search_in_vector_db(self, query, index, top_k=10):
        """Search for similar documents in the vector database"""
        if not self.documents_data:
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Perform similarity search
        scores, indices = index.search(query_embedding, top_k)
        
        # Return matching documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents_data):
                doc = self.documents_data[idx]
                results.append({
                    'score': float(score),
                    'document': doc,
                    'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                })
        
        return results


class ScanWorker(QThread):
    """Worker thread for scanning documents to prevent UI freezing"""
    
    progress_updated = Signal(str)
    scanning_finished = Signal()
    
    def __init__(self, processor, directory_path):
        super().__init__()
        self.processor = processor
        self.directory_path = directory_path
    
    def run(self):
        # Connect signals from processor to worker
        self.processor.progress_updated.connect(self.progress_updated.emit)
        self.processor.scanning_finished.connect(self.scanning_finished.emit)
        
        # Start scanning
        self.processor.scan_directory(self.directory_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"文档助手 v{__version__}")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize document processor
        self.processor = DocumentProcessor()
        self.vector_index = None
        self.db_path = "data/db/document_database.index"
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Directory selection section
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("文档目录：")
        self.dir_line_edit = QLineEdit()
        self.select_dir_button = QPushButton("选择目录")
        self.select_dir_button.clicked.connect(self.select_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_line_edit)
        dir_layout.addWidget(self.select_dir_button)
        main_layout.addLayout(dir_layout)
        
        # Scan button
        self.scan_button = QPushButton("扫描文档")
        self.scan_button.clicked.connect(self.start_scan)
        main_layout.addWidget(self.scan_button)
        
        # Progress label
        self.progress_label = QLabel("准备扫描文档...")
        main_layout.addWidget(self.progress_label)
        
        # Query section
        query_group = QWidget()
        query_layout = QVBoxLayout(query_group)
        
        query_title = QLabel("查询文档")
        query_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        query_layout.addWidget(query_title)
        
        # Query input
        self.query_text = QTextEdit()
        self.query_text.setMaximumHeight(100)
        query_layout.addWidget(QLabel("输入查询内容："))
        query_layout.addWidget(self.query_text)
        
        # Number of results input
        num_results_layout = QHBoxLayout()
        num_results_layout.addWidget(QLabel("显示数量："))
        self.num_results_spinbox = QSpinBox()
        self.num_results_spinbox.setValue(10)
        self.num_results_spinbox.setMinimum(1)
        self.num_results_spinbox.setMaximum(100)
        num_results_layout.addWidget(self.num_results_spinbox)
        num_results_layout.addStretch()  # Add stretch to align to left
        query_layout.addLayout(num_results_layout)
        
        # Query button
        self.query_button = QPushButton("搜索")
        self.query_button.clicked.connect(self.perform_query)
        query_layout.addWidget(self.query_button)
        
        main_layout.addWidget(query_group)
        
        # Add separator line between query area and results
        separator_line = self.create_separator_line()
        main_layout.addWidget(separator_line)
        
        # Results section
        results_title = QLabel("搜索结果")
        results_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(results_title)
        
        self.results_list = QListWidget()
        self.results_list.itemDoubleClicked.connect(self.open_result_file)
        main_layout.addWidget(self.results_list)
        
        # Check if database exists
        self.check_database_exists()
        
        # Worker thread for scanning
        self.scan_worker = None
    
    def select_directory(self):
        """Open dialog to select document directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Document Directory")
        if directory:
            self.dir_line_edit.setText(directory)
    
    def start_scan(self):
        """Start scanning documents in a separate thread"""
        directory = self.dir_line_edit.text().strip()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "警告", "请先选择一个有效目录。")
            return
        
        # Disable UI elements during scanning
        self.scan_button.setEnabled(False)
        self.query_button.setEnabled(False)
        self.progress_label.setText("正在扫描中...")
        
        # Create and start worker thread
        self.scan_worker = ScanWorker(self.processor, directory)
        self.scan_worker.progress_updated.connect(self.update_progress)
        self.scan_worker.scanning_finished.connect(self.on_scan_finished)
        self.scan_worker.start()
    
    def update_progress(self, message):
        """Update progress message"""
        self.progress_label.setText(message)
    
    def on_scan_finished(self):
        """Handle completion of scanning"""
        # Enable UI elements
        self.scan_button.setEnabled(True)
        self.query_button.setEnabled(True)
        
        # Save to vector database
        success = self.processor.save_to_vector_db(self.db_path)
        if success:
            self.progress_label.setText("扫描完成，数据库保存成功！")
            # Load the index for future queries
            self.vector_index = self.processor.load_from_vector_db(self.db_path)
        else:
            self.progress_label.setText("扫描完成但保存数据库失败。")
    
    def perform_query(self):
        """Perform similarity search on the vector database"""
        query_text = self.query_text.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "警告", "请先输入查询内容。")
            return
        
        if not self.vector_index:
            # Try to load the database
            self.vector_index = self.processor.load_from_vector_db(self.db_path)
        
        if not self.vector_index:
            QMessageBox.information(self, "信息", "未找到向量数据库。请先扫描文档。")
            return
        
        # Get number of results
        num_results = self.num_results_spinbox.value()
        
        # Perform search
        results = self.processor.search_in_vector_db(query_text, self.vector_index, num_results)
        
        # Display results
        self.display_results(results)
    
    def display_results(self, results):
        """Display search results in the list widget"""
        self.results_list.clear()
        
        # Remove duplicates by file path, keeping the one with the best score
        unique_results = {}
        for result in results:
            doc = result['document']
            file_path = doc['file_path']
            
            # If this file path hasn't been seen or has a better score than previous
            if file_path not in unique_results or result['score'] < unique_results[file_path]['score']:
                unique_results[file_path] = {
                    'result': result,
                    'score': result['score']
                }
        
        # Convert back to list and sort by score
        deduplicated_results = [item['result'] for item in unique_results.values()]
        deduplicated_results.sort(key=lambda x: x['score'])  # Sort by score ascending (better matches first)
        
        # Get the query text to highlight keywords
        query_text = self.query_text.toPlainText().strip()
        query_words = [word.strip() for word in query_text.split() if word.strip()]
        
        for result in deduplicated_results:
            doc = result['document']
            score = result['score']
            preview = result['content_preview']
            
            # Highlight keywords in the preview if query text exists (using plain text markers)
            if query_words:
                highlighted_preview = self.highlight_keywords_plain(preview, query_words)
            else:
                highlighted_preview = preview
            
            # Create the full text for the item (v1.0.4: only filename with complete path)
            item_text = f"{doc['original_filename']}\n完整路径: {doc['file_path']}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, doc['file_path'])  # Store file path for later use
            
            # Add comprehensive tooltip with all information for user convenience
            tooltip_text = f"文件名: {doc['original_filename']}\n完整路径: {doc['file_path']}\n相似度: {score:.2f}\n预览: {preview}"
            if query_words:
                # Add highlighted version to tooltip
                tooltip_text = self.create_highlighted_tooltip(tooltip_text, query_words)
            item.setToolTip(tooltip_text)
            
            self.results_list.addItem(item)
    
    def highlight_keywords_plain(self, text, keywords):
        """Highlight keywords in plain text by surrounding them with asterisks"""
        highlighted_text = text
        for keyword in keywords:
            # Replace keyword with marked version
            highlighted_text = highlighted_text.replace(
                keyword, 
                f"*{keyword}*"  # Surround with asterisks to indicate highlighting
            )
        return highlighted_text
    
    def create_highlighted_tooltip(self, text, keywords):
        """Create a tooltip with HTML-formatted highlighted keywords"""
        import html
        highlighted_text = html.escape(text)  # Escape HTML characters first
        
        for keyword in keywords:
            escaped_keyword = html.escape(keyword)
            # Replace keyword with highlighted HTML version for tooltip
            highlighted_text = highlighted_text.replace(
                escaped_keyword, 
                f'<b>{escaped_keyword}</b>'  # Bold for tooltip
            )
        return highlighted_text
    
    def create_separator_line(self):
        """Create a horizontal line separator"""
        line = QLabel()
        line.setFrameShape(QLabel.HLine)  # Horizontal line
        line.setFrameShadow(QLabel.Sunken)  # Sunken appearance
        line.setStyleSheet("border-top: 1px solid gray; margin: 5px 0px;")  # Styling
        return line
    
    def open_result_file(self, item):
        """Open the file associated with the selected result"""
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            # Open file with default application
            os.startfile(file_path)
        else:
            QMessageBox.warning(self, "错误", f"文件未找到: {file_path}")
    
    def check_database_exists(self):
        """Check if vector database exists and update UI accordingly"""
        if os.path.exists(self.db_path):
            self.vector_index = self.processor.load_from_vector_db(self.db_path)
            if self.vector_index:
                self.progress_label.setText("向量数据库已加载。准备就绪，可以搜索。")
            else:
                self.progress_label.setText("找到向量数据库文件但加载失败。")
        else:
            self.progress_label.setText("未找到向量数据库。请先扫描文档。")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()