# DocuGraph

## Overview

**DocuGraph** is a simple Python tool to explore and visualize the relationships within PDF documents. It extracts text from PDFs, tokenizes the text, generates embeddings, and then linearly steps over these tokens in the order they appear in the document to create a similarity-based directed graph with relationships based on the embedding similarity of these tokens.

## Features

- **PDF Parsing**: Extracts text from PDF files.
- **Text Embedding**: Converts text into numerical embeddings.
- **Graph Construction**: Builds a graph based on text similarity.
- **Visualization**: Generates visual representations of the text relationships.

### Limitations

- **No Graph Re-balancing**: The current implementation selects node embeddings on a first-come, first-serve basis. This means that nodes are not chosen as centroids or optimized points but are arbitrarily selected based on their distinctiveness at the time of insertion. As a result, some node embeddings may have stronger relationships with certain existing nodes in the graph at the end of the construction process.

## Installation

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

### Running

Execute the main script to analyze a PDF document:

```bash
python src/main.py
```

Make sure you have a PDF file at the specified path or modify `main.py` to point to your PDF:

```python
pdf_bytes = load_pdf_bytes("../data/input/your_document.pdf")
```

### Configuration

Adjust processing parameters by modifying `src/_util/_config.py`:

```python
SIMILARITY_THRESHOLD = 0.82  # Cosine similarity threshold for embeddings
WHITESPACE_THRESHOLD = 0.6  # Threshold for whitespace node detection
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
└── README.md
```

- **src/**: Source code directory.
  - **_util/**: Utility modules for configuration and common functions.
  - **backend/**: Modules handling graph operations.
  - `main.py`: Entry point of the application.
- **data/**: Directory to store input PDF files.
- **output/**: Directory where output files like graph visualizations are saved.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: Project documentation.
## Example

After setting up, run the program to generate a graph visualization of your PDF:

```bash
cd src
python main.py
```

The visualization will be saved in the `output/` directory as `graph_visualization.png`.

Edge weights between two nodes represent the frequency we move from one node to the next.
![see `examples/data/poem.pdf`](examples/outputs/graph_visualization_poem.png)
![see `examples/data/resume.pdf`](examples/outputs/graph_visualization_resume.png)
`I've also visualized Fahrenheit 451 and some Q-10 filings, but the charts are too big for github`