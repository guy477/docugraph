# ______________________________________________________________________________________#
# _________________________________ CONSTANTS __________________________________________#
# ______________________________________________________________________________________#

# `SIMILARITY_THRESHOLD` determines the threshold for cosine similarity.
# This value can vary based on the embedding model in use.
# For instance, with `text-embedding-ada-002`, graph interconnectivity becomes apparent
# when `SIMILARITY_THRESHOLD` is set between 0.785 and 0.790 [found experimentally].
# NOTE: 0 means all embeddings are identical; 1 means all embeddings are completely unique.
SIMILARITY_THRESHOLD = .7875

# `WHITESPACE_THRESHOLD` defines the ratio of whitespace tokens to total tokens in a node,
# beyond which the node is considered as `whitespace`.
# NOTE: This threshold is primarily used for visualization purposes.
WHITESPACE_THRESHOLD = 0.9