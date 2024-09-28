import asyncio
import numpy as np
import pdfplumber
from io import BytesIO
import logging
from typing import List, Dict, Optional, Tuple, Set
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import hashlib
import json
import re
from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms import community
from collections import defaultdict
import math

from _util._config import *



# Configure logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG level for detailed logs
logger = logging.getLogger(__name__)


# Assume we have an OpenAI API key set in the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to generate embeddings asynchronously
async def get_embeddings(text_list: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI API, optimizing by removing duplicates.

    Args:
        text_list (List[str]): List of text strings to embed.

    Returns:
        np.ndarray: Array of embeddings corresponding to the original text_list order.
    """
    logger.info("get_embeddings called with %d texts", len(text_list))
    try:
        # Remove duplicates while preserving order
        unique_texts = []
        seen_texts = set()
        for text in text_list:
            if text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append(text)
        
        text_to_embedding = {}
        batch_size = 2048  # Adjust based on API limits and performance

        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Generating embeddings"):
            batch = unique_texts[i:i + batch_size]
            logger.debug("Generating embeddings for unique batch %d to %d", i, i + len(batch))
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            ).to_dict()
            batch_embeddings = [np.array(data['embedding']) for data in response['data']]
            for text, embedding in zip(batch, batch_embeddings):
                text_to_embedding[text] = embedding

        # Map embeddings back to the original text_list order
        embeddings = [text_to_embedding[text] for text in text_list]
        logger.info("get_embeddings completed successfully with optimized API calls")
        return np.array(embeddings)
    except Exception as e:
        logger.error("Error during embedding call: %s", str(e), exc_info=True)
        return np.array([])

# Function to parse PDF content
def parse_pdf_content(pdf_bytes: bytes) -> str:
    """
    Parse PDF content and extract text.

    Args:
        pdf_bytes (bytes): The bytes of the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    logger.debug("Starting PDF parsing.")
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
            logger.info(f"Number of pages in PDF: {num_pages}")
            text = ''
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'  # Add space to separate pages
                    logger.debug(f"Extracted text from page {i}.")
                else:
                    logger.warning(f"No text found on page {i}.")
        logger.info("Completed PDF parsing.")
        return text
    except Exception as e:
        logger.error(f"Error while parsing PDF: {e}")
        raise

# Function to tokenize text
def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into tokens while preserving spaces.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: The list of tokens, including spaces.
    """
    # Split the text into words and spaces
    tokens = re.split(r'(\s+)', text)
    
    # Filter out any empty strings
    tokens = [token for token in tokens if token]
    
    logger.debug(f"Tokenized text into {len(tokens)} tokens with spaces preserved.")
    return tokens


def load_pdf_bytes(file_path):
    """
    Loads a PDF file and returns its raw bytes.

    Parameters:
        file_path (str): The path to the PDF file.

    Returns:
        bytes: The raw bytes of the PDF file.
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_bytes = file.read()
            logging.info(f"PDF bytes loaded successfully: {file_path}")
            return pdf_bytes
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the PDF bytes: {e}")
        raise
