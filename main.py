import sys
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QTextEdit, QLabel, QLineEdit, QFileDialog, 
                              QMessageBox, QSpinBox, QListWidget, QListWidgetItem, QTextBrowser,
                              QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon, QTextCursor
import PyPDF2
from docx import Document
import pickle
from version import __version__
import logging
from datetime import datetime
import traceback
from plugins import get_plugin_manager

# Fix for PyInstaller compatibility with transformers library
if getattr(sys, 'frozen', False):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for transformers logging issue when stdout is not available
import sys
_original_stdout = sys.stdout

# Override stdout temporarily during imports if needed
class DummyStdout:
    def write(self, txt):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False

# Temporarily replace stdout if it's not available during initialization
if not hasattr(sys.stdout, 'isatty'):
    sys.stdout = DummyStdout()


class DocumentProcessor(QObject):
    """Handles document processing and vector database operations"""
    
    # Signals for communication with GUI
    progress_updated = Signal(str)
    scanning_finished = Signal()
    progress_value = Signal(int)  # Signal to update progress bar value
    
    def __init__(self):
        super().__init__()
        # Initialize the model with proper environment settings for PyInstaller
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Handle the stdout issue during model loading
        import sys
        original_stdout = sys.stdout
        if not hasattr(sys.stdout, 'isatty'):
            class DummyStdout:
                def write(self, txt):
                    pass
                def flush(self):
                    pass
                def isatty(self):
                    return False
            sys.stdout = DummyStdout()
        
        try:
            # Try to load model from local directory first
            local_model_path = "model/all-MiniLM-L6-v2"
            if os.path.exists(local_model_path):
                # Load from local directory
                self.model = SentenceTransformer(local_model_path, device='cpu')
            else:
                # Load from online (this will download and cache for future use)
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        finally:
            # Restore original stdout after model loading
            sys.stdout = original_stdout
        
        self.documents_data = []  # Store document content and metadata
        self.plugin_manager = get_plugin_manager()  # Initialize plugin manager
        
        # Cache directory for storing scanned files info
        self.cache_dir = "data/cache"
        self.cache_file = os.path.join(self.cache_dir, "file_cache.pkl")
        
        # Load existing cache
        self.file_cache = self.load_cache()
        
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
    
    def get_file_info(self, file_path):
        """Get file information for caching purposes"""
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'path': file_path
        }
    
    def load_cache(self):
        """Load cached file information from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def save_cache(self, cache_data):
        """Save file cache information to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def is_file_cached_and_valid(self, file_path, cache_data):
        """Check if file is in cache and hasn't changed"""
        if file_path not in cache_data:
            return False
        
        current_info = self.get_file_info(file_path)
        cached_info = cache_data[file_path]
        
        # Compare file size and modification time
        return (current_info['size'] == cached_info['size'] and 
                abs(current_info['mtime'] - cached_info['mtime']) < 1)  # 1 second tolerance
    
    def scan_directory(self, directory_path, worker_thread=None, full_scan=False):
        """Scan directory for supported documents and extract content"""
        self.progress_updated.emit("开始扫描目录...")
        
        # Load cache at the beginning
        cache_data = self.load_cache() if not full_scan else {}
        
        # Track stats for display
        total_size = 0
        start_time = time.time()
        
        # Supported extensions
        supported_ext = ['.txt', '.pdf', '.doc', '.docx', '.epub']
        
        # Find all supported files
        files = []
        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_ext:
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
                    total_size += os.path.getsize(file_path)
        
        total_files = len(files)
        self.progress_updated.emit(f"找到 {total_files} 个待处理文件，总大小: {total_size / (1024*1024):.2f} MB...")
        
        if total_files == 0:
            self.progress_updated.emit("未找到支持的文档文件。")
            self.scanning_finished.emit()
            return
        
        # Process each file
        self.documents_data = []
        
        # Determine which files need to be processed (based on cache)
        files_to_process = []
        if full_scan:
            files_to_process = files  # Process all files if full scan
            cache_data = {}  # Clear cache for full scan
        else:
            for file_path in files:
                if not self.is_file_cached_and_valid(file_path, cache_data):
                    files_to_process.append(file_path)
                else:
                    # If file is cached, we could potentially load from cache here
                    # For now, we'll just note that it was skipped
                    pass
        
        files_to_process_count = len(files_to_process)
        if files_to_process_count < total_files:
            self.progress_updated.emit(f"缓存命中: {total_files - files_to_process_count} 个文件已缓存，只需处理 {files_to_process_count} 个新/修改文件...")
        
        # Process files that need processing
        for i, file_path in enumerate(files_to_process):
            # Check if paused
            if worker_thread and worker_thread.is_paused():
                while worker_thread.is_paused() and not worker_thread.is_stopped():
                    time.sleep(0.1)  # Small delay to reduce CPU usage
            
            # Check if stopped
            if worker_thread and worker_thread.is_stopped():
                self.progress_updated.emit("扫描已停止。")
                return
            
            self.progress_updated.emit(f"正在处理文件 {i+1}/{files_to_process_count}: {os.path.basename(file_path)}")
            
            # Update progress bar
            progress_percent = int((i + 1) / files_to_process_count * 100) if files_to_process_count > 0 else 100
            if worker_thread:
                worker_thread.progress_value.emit(progress_percent)
            
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
            
            # Update cache with processed file info
            cache_data[file_path] = self.get_file_info(file_path)
        
        # Save updated cache
        self.save_cache(cache_data)
        
        elapsed_time = time.time() - start_time
        self.progress_updated.emit(f"已处理来自 {total_files} 个文件的 {len(self.documents_data)} 个内容块。总大小: {total_size / (1024*1024):.2f} MB，耗时: {elapsed_time:.2f}秒。")
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
        """Save the vector database to disk"""
        try:
            # Create directory if it doesn't exist
            db_dir = os.path.dirname(db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Emit progress update
            if hasattr(self, 'progress_updated'):
                self.progress_updated.emit("开始编码文档内容为向量...")
            
            # Encode all document content to vectors
            documents = [item['content'] for item in self.documents_data]
            
            # Emit progress update
            if hasattr(self, 'progress_updated'):
                self.progress_updated.emit(f"正在为 {len(documents)} 个文档块生成向量表示...")
            
            # Encode documents in batches to provide progress updates
            embeddings = self.model.encode(documents, show_progress_bar=False)  # We'll handle progress ourselves
            
            # Emit progress update
            if hasattr(self, 'progress_updated'):
                self.progress_updated.emit("向量编码完成，正在构建索引...")
            
            # Create FAISS index
            dimension = embeddings.shape[1]  # Get embedding dimension
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            # Emit progress update
            if hasattr(self, 'progress_updated'):
                self.progress_updated.emit("正在保存向量数据库...")
            
            # Save index and document data
            faiss.write_index(index, db_path)
            
            # Save document metadata separately
            metadata_path = db_path.replace('.index', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.documents_data, f)
            
            # Emit final progress update
            if hasattr(self, 'progress_updated'):
                self.progress_updated.emit("向量数据库保存完成！")
            
            return True
        except Exception as e:
            error_msg = f"Error saving vector database: {e}"
            print(error_msg)
            # If MainWindow is available, log the exception
            try:
                from main import logging
                import traceback
                logging.error(f"save_to_vector_db - 异常: {str(e)}\n{traceback.format_exc()}")
            except:
                pass  # If logging is not available, just print
            return False
    
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
            error_msg = f"Error loading vector database: {e}"
            print(error_msg)
            # If MainWindow is available, log the exception
            try:
                from main import logging
                import traceback
                logging.error(f"load_from_vector_db - 异常: {str(e)}\n{traceback.format_exc()}")
            except:
                pass  # If logging is not available, just print
            return None
    
    def search_similar_documents(self, query, top_k=10, index=None):
        """Search for similar documents in the vector database"""
        try:
            if not self.documents_data or index is None:
                return []
            
            # Generate embedding for the query
            query_embedding = self.model.encode([query])
            
            # Normalize query embedding for cosine similarity
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Perform similarity search
            scores, indices = index.search(query_embedding, top_k)
            
            # Return matching documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents_data) and idx >= 0:
                    doc = self.documents_data[idx]
                    results.append({
                        'score': float(score),
                        'file_path': doc['file_path'],
                        'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    })
            
            return results
        except Exception as e:
            error_msg = f"Error searching vector database: {e}"
            print(error_msg)
            # If MainWindow is available, log the exception
            try:
                from main import logging
                import traceback
                logging.error(f"search_similar_documents - 异常: {str(e)}\n{traceback.format_exc()}")
            except:
                pass  # If logging is not available, just print
            return []
    
    def search_in_vector_db(self, query, index, top_k=10):
        """Search for similar documents in the vector database"""
        try:
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
        except Exception as e:
            error_msg = f"Error searching vector database: {e}"
            print(error_msg)
            # If MainWindow is available, log the exception
            try:
                from main import logging
                import traceback
                logging.error(f"search_in_vector_db - 异常: {str(e)}\n{traceback.format_exc()}")
            except:
                pass  # If logging is not available, just print
            return []


class ScanWorker(QThread):
    """Worker thread for scanning documents to prevent UI freezing"""
    
    progress_updated = Signal(str)
    scanning_finished = Signal()
    progress_value = Signal(int)  # Signal to update progress bar value
    
    def __init__(self, processor, directory_path, full_scan=False):
        super().__init__()
        self.processor = processor
        self.directory_path = directory_path
        self.full_scan = full_scan
        self._paused = False
        self._stopped = False
    
    def run(self):
        # Connect signals from processor to worker
        self.processor.progress_updated.connect(self.progress_updated.emit)
        self.processor.progress_value.connect(self.progress_value.emit)
        self.processor.scanning_finished.connect(self.scanning_finished.emit)
        
        # Start scanning
        self.processor.scan_directory(self.directory_path, self, self.full_scan)
    
    def pause(self):
        self._paused = True
    
    def resume(self):
        self._paused = False
    
    def stop(self):
        self._stopped = True
    
    def is_paused(self):
        return self._paused
    
    def is_stopped(self):
        return self._stopped


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"文档助手 v{__version__}")
        self.setGeometry(100, 100, 1000, 700)
        
        # Setup logging first
        self.setup_logging()
        
        # Set application icon
        self.set_app_icon()
        
        # Create data and db directories if they don't exist
        self.ensure_directories_exist()
        
        # Initialize document processor
        self.processor = DocumentProcessor()
        self.vector_index = None
        self.db_path = "data/db/document_database.index"
        
        # Setup basic UI first (without loading database)
        self.setup_basic_ui()
        
        # Load database asynchronously
        self.load_database_async()
    
    def setup_basic_ui(self):
        """Setup the basic UI elements without loading database"""
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
        
        # Scan and query buttons
        button_layout = QHBoxLayout()
        self.scan_button = QPushButton("扫描文档")
        self.scan_button.clicked.connect(self.start_scan)
        button_layout.addWidget(self.scan_button)
        
        # Full scan button
        self.full_scan_button = QPushButton("全量扫描")
        self.full_scan_button.clicked.connect(lambda: self.start_scan(full_scan=True))
        button_layout.addWidget(self.full_scan_button)
        
        # Pause/Resume button
        self.pause_resume_button = QPushButton("暂停")
        self.pause_resume_button.clicked.connect(self.toggle_pause_resume)
        self.pause_resume_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.pause_resume_button)
        
        # Stop button
        self.stop_button = QPushButton("终止")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.stop_button)
        
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
        
        main_layout.addLayout(button_layout)  # Add buttons layout first
        
        # Progress label for status updates
        self.progress_label = QLabel("正在初始化...")
        main_layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Initially hidden
        main_layout.addWidget(self.progress_bar)
        
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
        
        # Worker thread for scanning
        self.scan_worker = None
    
    def load_database_async(self):
        """Load database asynchronously in a separate thread"""
        # Start database loading in a thread
        db_thread = threading.Thread(target=self._load_database_in_background)
        db_thread.daemon = True  # Allow the thread to be terminated when main program exits
        db_thread.start()
    
    def _load_database_in_background(self):
        """Background method to load the database"""
        # Check if database exists and load it
        if os.path.exists(self.db_path):
            try:
                self.vector_index = self.processor.load_from_vector_db(self.db_path)
                # Update UI in the main thread
                if self.vector_index:
                    # Use QTimer to call the UI update method in the main thread
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._update_status_loaded_successfully())
                else:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self._update_status_load_failed("数据库文件格式错误"))
            except Exception as e:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._update_status_load_failed(f"数据库加载失败: {str(e)}"))
        else:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._update_status_no_database())
    
    def _update_status_loaded_successfully(self):
        """Update UI status when database loads successfully - called from main thread"""
        self.progress_label.setText("向量数据库已加载。准备就绪，可以搜索。")
    
    def _update_status_load_failed(self, error_msg):
        """Update UI status when database fails to load - called from main thread"""
        self.progress_label.setText(error_msg)
    
    def _update_status_no_database(self):
        """Update UI status when no database is found - called from main thread"""
        self.progress_label.setText("未找到向量数据库。请先扫描文档。")
    
    def select_directory(self):
        """Open dialog to select document directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Document Directory")
        if directory:
            self.dir_line_edit.setText(directory)
    
    def start_scan(self, full_scan=False):
        """Start scanning documents in a separate thread"""
        try:
            directory = self.dir_line_edit.text().strip()
            if not directory or not os.path.exists(directory):
                QMessageBox.warning(self, "警告", "请先选择一个有效目录。")
                return
            
            # Disable UI elements during scanning
            self.scan_button.setEnabled(False)
            self.full_scan_button.setEnabled(False)  # Disable full scan button too
            self.query_button.setEnabled(False)
            self.progress_label.setText("正在扫描中..." if not full_scan else "正在进行全量扫描...")
            
            # Show progress bar and reset
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Create and start worker thread
            self.scan_worker = ScanWorker(self.processor, directory, full_scan)
            self.scan_worker.progress_updated.connect(self.update_progress)
            self.scan_worker.progress_value.connect(self.update_progress_bar)
            self.scan_worker.scanning_finished.connect(self.on_scan_finished)
            
            # Enable pause/resume button
            self.pause_resume_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.pause_resume_button.setText("暂停")
            
            self.scan_worker.start()
        except Exception as e:
            self.log_exception(e, "start_scan")
            QMessageBox.critical(self, "错误", f"启动扫描时发生错误: {str(e)}")
    
    def toggle_pause_resume(self):
        """Toggle between pause and resume states"""
        try:
            if self.scan_worker:
                if self.scan_worker.is_paused():
                    # Resume
                    self.scan_worker.resume()
                    self.pause_resume_button.setText("暂停")
                    self.progress_label.setText("正在恢复扫描...")
                else:
                    # Pause
                    self.scan_worker.pause()
                    self.pause_resume_button.setText("继续")
                    self.progress_label.setText("扫描已暂停，点击继续按钮恢复...")
        except Exception as e:
            self.log_exception(e, "toggle_pause_resume")
            QMessageBox.critical(self, "错误", f"切换暂停/继续状态时发生错误: {str(e)}")
    
    def stop_scan(self):
        """Stop the scanning process"""
        try:
            if self.scan_worker:
                self.scan_worker.stop()
                self.progress_label.setText("正在停止扫描...")
                # Wait briefly for thread to stop
                time.sleep(0.2)  # Brief wait to allow thread to process stop request
                self.progress_label.setText("扫描已终止。")
                
                # Reset UI elements
                self.scan_button.setEnabled(True)
                self.full_scan_button.setEnabled(True)  # Re-enable full scan button
                self.query_button.setEnabled(True)
                self.pause_resume_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.pause_resume_button.setText("暂停")
                
                # Hide progress bar
                self.progress_bar.setVisible(False)
                
                # Reset worker reference
                self.scan_worker = None
        except Exception as e:
            self.log_exception(e, "stop_scan")
            QMessageBox.critical(self, "错误", f"停止扫描时发生错误: {str(e)}")
    
    def update_progress_bar(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def update_progress(self, message):
        """Update progress message"""
        self.progress_label.setText(message)
    
    def on_scan_finished(self):
        """Handle completion of scanning"""
        try:
            # Enable UI elements
            self.scan_button.setEnabled(True)
            self.full_scan_button.setEnabled(True)  # Re-enable full scan button
            self.query_button.setEnabled(True)
            self.pause_resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)  # Also disable stop button
            self.pause_resume_button.setText("暂停")
            
            # Hide progress bar
            self.progress_bar.setVisible(False)
            
            # Save to vector database
            success = self.processor.save_to_vector_db(self.db_path)
            if success:
                self.progress_label.setText("扫描完成，数据库保存成功！")
                # Load the index for future queries
                self.vector_index = self.processor.load_from_vector_db(self.db_path)
            else:
                self.progress_label.setText("扫描完成但保存数据库失败。")
        except Exception as e:
            self.log_exception(e, "on_scan_finished")
            self.progress_label.setText(f"扫描完成但处理结果时发生错误: {str(e)}")
    
    def perform_query(self):
        """Perform similarity search on the vector database"""
        try:
            query_text = self.query_text.toPlainText().strip()
            if not query_text:
                QMessageBox.warning(self, "警告", "请先输入查询内容。")
                return
            
            num_results = self.num_results_spinbox.value()
            
            if not self.vector_index:
                if os.path.exists(self.db_path):
                    # Load index if not already loaded
                    self.vector_index = self.processor.load_from_vector_db(self.db_path)
                    if not self.vector_index:
                        QMessageBox.warning(self, "警告", "向量数据库加载失败，请重新扫描文档。")
                        return
                else:
                    QMessageBox.warning(self, "警告", "未找到向量数据库，请先扫描文档。")
                    return
            
            # Perform query
            results = self.processor.search_similar_documents(query_text, num_results, self.vector_index)
            
            # Update results list
            self.results_list.clear()
            unique_files = set()
            for result in results:
                file_path = result['file_path']
                if file_path not in unique_files:
                    item = QListWidgetItem(file_path)
                    self.results_list.addItem(item)
                    unique_files.add(file_path)
        except Exception as e:
            self.log_exception(e, "perform_query")
            QMessageBox.critical(self, "错误", f"执行查询时发生错误: {str(e)}")
    
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
    
    def ensure_directories_exist(self):
        """Create data, db, model, log, and cache directories if they don't exist"""
        data_dir = "data"
        db_dir = os.path.join(data_dir, "db")
        model_dir = "model"
        log_dir = "log"
        cache_dir = os.path.join(data_dir, "cache")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        # Create db directory if it doesn't exist
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging to record exceptions to daily log files"""
        # Create log directory if it doesn't exist
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"log-{today}.txt")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        # Log application start
        logging.info(f"文档助手 v{__version__} 启动")
    
    def log_exception(self, e, context="General"):
        """Log exception with timestamp and context"""
        error_msg = f"{context} - 异常: {str(e)}"
        error_traceback = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_traceback}")
    
    def set_app_icon(self):
        """Set the application icon from 128.png file"""
        import os
        from PySide6.QtGui import QIcon
        
        icon_path = "128.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
    
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
    
    # Create and show splash screen
    from PySide6.QtWidgets import QSplashScreen
    from PySide6.QtGui import QPixmap
    import os
    
    splash = None
    if os.path.exists("128.png"):
        pixmap = QPixmap("128.png")
        if not pixmap.isNull():
            splash = QSplashScreen(pixmap)
            splash.show()
            app.processEvents()  # Allow the splash screen to be displayed
    
    # Create main window
    window = MainWindow()
    
    # Hide splash screen when main window is ready
    if splash:
        splash.finish(window)
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()