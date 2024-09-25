from _util._utils import *
from backend.node import *
from backend.graph_visualizer import *

async def main(pdf_bytes: bytes):
    # Step 1: Parse PDF and extract text
    text = parse_pdf_content(pdf_bytes)
    logger.debug(f"Extracted Text Sample: {text[:500]}...")  # Log first 500 characters

    # Handle edge case where no text is extracted
    if not text:
        logger.error("No text could be extracted from the PDF.")
        return

    # Step 2: Tokenize the text
    tokens = tokenize_text(text)
    tokens = tokens
    logger.debug(f"Total Tokens: {len(tokens)}")
    logger.debug(f"Tokens Sample: {tokens[:20]}...")  # Log first 20 tokens

    # Handle edge case where tokenization results in no tokens
    if not tokens:
        logger.error("Tokenization resulted in zero tokens.")
        return

    # Step 3: Construct the embedding graph
    root_node, index_to_node = await construct_embedding_graph(tokens, SIMILARITY_THRESHOLD)

    if VALIDATE_RECONSTRUCTION:
        # Step 4: Reconstruct the document from the graph
        reconstructed_text = reconstruct_document(index_to_node, len(tokens))
        logger.debug(f"Reconstructed Text Sample: {reconstructed_text[:500]}...")  # Log first 500 characters

    # Step 5: Verify reconstruction
    if VALIDATE_RECONSTRUCTION and reconstructed_text == text:
        logger.info("Document reconstruction successful and matches the original text.")
    elif VALIDATE_RECONSTRUCTION:
        logger.warning("Document reconstruction does not match the original text.")
        # Output differences for analysis
        logger.debug(f"Original Text:\n{text}")
        logger.debug(f"Reconstructed Text:\n{reconstructed_text}")

    visualizer = GraphVisualizer(index_to_node, len(tokens))
    visualizer.visualize('../output/graph_visualization.png')

# Example usage
if __name__ == "__main__":
    # Load your PDF bytes here
    # For demonstration, we will assume pdf_bytes is available
    # Replace 'path_to_pdf' with your actual PDF file path
    pdf_bytes = load_pdf_bytes("../data/resume_sample.pdf")

    # Since we cannot actually call the OpenAI API in this environment,
    # and we're simulating embeddings, the code execution is illustrative only.

    # To run the code, uncomment the following lines and provide the pdf_bytes variable
    asyncio.run(main(pdf_bytes))
