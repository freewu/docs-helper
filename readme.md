# Document Helper Application

A PySide6-based application for scanning and querying documents using vector databases.

## Features

- Scan documents (DOCX, DOC, TXT, PDF) in a selected directory
- Extract text content and store in a vector database
- Query documents using semantic search
- Display and open matching documents

## Requirements

- Python 3.8+
- PySide6
- PyPDF2
- python-docx
- sentence-transformers
- faiss-cpu
- numpy

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python main.py
   ```

## Usage

1. Select a directory containing documents to scan
2. Click "Scan Documents" to process all supported files in the directory
3. Enter your query in the text box
4. Adjust the number of results to display (default: 10)
5. Click "Search" to find relevant documents
6. Double-click on results to open the corresponding file

## Building Executable

To build a standalone executable:
```
python package.py
```

The executable will be created in the `dist` directory with version information included.

## Project Structure

- `main.py`: Main application code
- `package.py`: Packaging script
- `requirements.txt`: Dependencies
- `version.py`: Version information
- `data/`: Directory for vector database storage
