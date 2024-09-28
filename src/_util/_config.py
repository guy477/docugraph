# ______________________________________________________________________________________#
# _________________________________ CONSTANTS __________________________________________#
# ______________________________________________________________________________________#

# `PDF_LOCATION` is the location of the PDF file to be processed.
PDF_LOCATION = "../data/input/fahrenheit_451.pdf"

# `CHART_SAVE_PATH` is the location where the charts will be saved.
CHART_SAVE_PATH = "../data/output/"

# `SIMILARITY_THRESHOLD` determines the threshold for cosine similarity.
# This value can vary based on the embedding model in use.
# For instance, with `text-embedding-ada-002`, graph interconnectivity becomes apparent
# when `SIMILARITY_THRESHOLD` is set between 0.785 and 0.790 [found experimentally].
# NOTE: 0 means all embeddings are identical; 1 means all embeddings are unique.
SIMILARITY_THRESHOLD = .82

# `WHITESPACE_THRESHOLD` defines the ratio of whitespace tokens to total tokens in a node,
# beyond which the node is considered as `whitespace`.
WHITESPACE_THRESHOLD = 0.6

# TOPIC_WINDOW_SIZE
TOPIC_WINDOW_SIZE = 30 # divide this number by 2 to get a word-count estimate

# REDUNDANT_TOKEN_THRESHOLD
REDUNDANT_TOKEN_THRESHOLD = 0.9999 # Reduces noise in the output. Intended to remove common, non-informative tokens (like `and`, `those`, etc.).

# MERGE_SIMILAR_TOPICS_THRESHOLD
MERGE_SIMILAR_TOPICS_THRESHOLD = 0.8

# If True, show the actual text for the topic's index.
# If False, show the node's `token_text` for the topic's index.
USE_ACTUAL_TEXT = True

# `VALIDATE_RECONSTRUCTION` determines whether or not to reconstruct the document from the graph.
VALIDATE_RECONSTRUCTION = False