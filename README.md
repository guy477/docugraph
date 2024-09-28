# DocuGraph

## Overview

**DocuGraph** is a simple Python tool to explore and visualize the relationships within PDF documents. It extracts text from PDFs, tokenizes the text, generates embeddings, and then linearly steps over these tokens in the order they appear in the document to create a similarity-based directed graph with relationships based on the embedding similarity of these tokens.

## Features

- **PDF Parsing**: Extracts text from PDF files.
- **Text Embedding**: Converts text into vector embeddings.
- **Graph Construction**: Builds a graph based on text similarity. Graph construction assumptions ensure every token in the document is assigned to its most similar Node after the first pass.
- **Topic/Theme Extraction**: Identifies and extracts major topics/themes from the graph (seems to do something meaningful, but I'm not sure of its robustness).
- **Visualization**: Generates visual representations of the text relationships.

## Limitations

- --**No Cluster Balancing**: The current implementation `clusters` Nodes on a first-come, first-serve basis. This means that a Node's embedding is not chosen as a centroids or optimized clusters and is instead arbitrarily selected based on their distinctiveness at the time of insertion to the graph.
- **No Sub-graph Analysis**: Identifying and analyzing the substructures of the graphs, along with how we move between these substructues over time, is not supported.
- **PDF Text Extraction**: The current implementation uses `pdfplumber` to extract text from PDFs. This works well for most technical documents, but may not work as well for PDFs with complex or unstandardized formatting.

## Graph Construction Assumptions

Our project constructs a graph from document tokens using their embeddings and similarity measures. The following assumptions underlie the graph construction process:

1. **Sequential Token Processing**

   - Tokens from the document are processed in a linear sequence.
   - For each token, an embedding vector is calculated to represent its semantic meaning.

2. **Similarity-Based Node Assignment**

   - For each token's embedding, we check if there is an existing Node in the graph within a predefined similarity threshold.
   - If such a Node exists, the token is added to that Node, grouping similar tokens together.

3. **Creation of Unique Nodes**

   - If no existing Node meets the similarity criterion for the current token, a new Node is created.
   - This ensures that each Node has an embedding distinct from all other Nodes in the graph.

4. **Maintaining Token Relationships**

   - The Node corresponding to the previous token is linked to the current token's Node as a child, preserving the sequential relationship.
   - This linkage constructs a graph that reflects both the semantic similarity and the order of tokens in the document.

5. **Optimal Node Assignment**

   - By the end of the graph construction, each token is guaranteed to be assigned to the most appropriate Node based on the similarity threshold.

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
WHITESPACE_THRESHOLD = 0.6  # Threshold for whitespace Node detection
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
│   │   └── Node.py
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

### Sample Topics/Themes (Extracted from `fahrenheit_451.pdf`)
```
(Topic Frequency: 6)  and the entire structure of hotel blasted down upon her, carrying her with a million pounds as it
(Topic Frequency: 6)  her with a million pounds of brick, metal, plaster, and wood, to meet other people in the hives below, all
(Topic Frequency: 6)  on their quick way down to the cellar where explosion rid itself of them in its own unreasonable way.
(Topic Frequency: 6)  the cellar where explosion rid itself of them in its own unreasonable way. I remember. Montag clung to
 .
 .
 .
(Topic Frequency: 13)  eye, of fur and muzzle hoof, he was a thing horn blood that the fire. He brush liquid
(Topic Frequency: 13)  Schopenhauer, and this one is Einstein, here at my elbow Mr. Albert Schweitzer, a very kind philosopher indeed. Here we all are, Montag.
(Topic Frequency: 13)  down to the minimum !" he yelled: "What?" she cried. "Keep it fifty-five, minimum! " shouted. "The
(Topic Frequency: 14)  drop by drop, there is at last a which makes it run over, so in series of kindnesses one the heart over.'"
(Topic Frequency: 14)  with you! We're twins, we're not alone any more, separated out in different parlours, no contact between. If you need help when Beatty pries at
(Topic Frequency: 19)  fire, One, Mildred, two, Clarisse. three, four, five, Clarisse, uncle, sleeping-tablets, men, disposable tissue, coat-tails, blow, wad, flush, tablets, tissues, flush.
```

### Example Visualizations
The visualization will be saved in the `output/` directory as `graph_visualization.png`.

Edge weights between two Nodes represent the frequency we move from one Node to the next.
![see `examples/data/poem.pdf`](examples/outputs/graph_visualization_poem.png)
![see `examples/data/resume.pdf`](examples/outputs/graph_visualization_resume.png)
`I've also visualized Fahrenheit 451 and some Q-10 filings, but the charts are too big for github`