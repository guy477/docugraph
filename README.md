# DocuGraph

## Overview

**DocuGraph** is a simple Python tool I built in a day to explore and visualize the relationships within PDF documents. It converts text from PDFs into embeddings, creates a similarity-based graph, and visualizes the connections between different text segments. It's a fun side project that showcases how text can be represented and analyzed using basic natural language processing techniques.

## Features

- **PDF Parsing**: Extracts text from PDF files.
- **Text Embedding**: Converts text into numerical embeddings.
- **Graph Construction**: Builds a graph based on text similarity.
- **Visualization**: Generates visual representations of the text relationships.

## Installation

### Prerequisites

- **Python 3.11** (This is the version I use.)
- [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.
- [OpenAI API Key](https://platform.openai.com/account/api-keys)

### Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/guy477/docugraph.git
    cd docugraph
    ```

2. **Create a Conda Environment**

    ```bash
    conda create -n docugraph_env python=3.11
    conda activate docugraph_env
    ```

3. **Install Dependencies**

    You can install the necessary packages using `pip` within the activated Conda environment:

    ```bash
    pip install asyncio numpy pdfplumber openai scikit-learn matplotlib networkx pygraphviz
    ```

4. **Configure Environment Variables**

    Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
    ```

## Usage

### Running the Processor

Execute the main script to analyze a PDF document:

```bash
python src/main.py
```

Make sure you have a PDF file at the specified path or modify `main.py` to point to your PDF:

```python
pdf_bytes = load_pdf_bytes("../data/your_document.pdf")
```

### Configuration

Adjust processing parameters by modifying `src/_util/_config.py`:

```python
SIMILARITY_THRESHOLD = 0.8  # Cosine similarity threshold for embeddings
WHITESPACE_THRESHOLD = 0.8  # Threshold for whitespace node detection
```

## Project Structure

```plaintext
docugraph/
├── src/
│   ├── _util/
│   │   ├── _config.py
│   │   └── _utils.py
│   ├── backend/
│   │   ├── graph_visualizer.py
│   │   └── node.py
│   └── main.py
├── data/
│   └── your_document.pdf
├── output/
│   └── graph_visualization.png
├── .gitignore
├── README.md
└── environment.yml
```

- **src/**: Source code directory.
  - **_util/**: Utility modules for configuration and common functions.
  - **backend/**: Modules handling graph operations.
  - `main.py`: Entry point of the application.
- **data/**: Directory to store input PDF files.
- **output/**: Directory where output files like graph visualizations are saved.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: Project documentation.
- **environment.yml**: Conda environment configuration file.

## Example

After setting up, run the processor to generate a graph visualization of your PDF:

```bash
python src/main.py
```

The visualization will be saved in the `output/` directory as `graph_visualization.png`.

![Semantic Graph of Resume](output/graph_visualization.png)