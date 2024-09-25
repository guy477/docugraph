from _util._utils import *

class Node:
    def __init__(self, embedding: Optional[np.ndarray], token_index: int, token_text: str):
        """
        Initialize a Node with an embedding, token index, and token text.

        Args:
            embedding (Optional[np.ndarray]): The embedding vector for this node.
            token_index (int): The index of the token in the document.
            token_text (str): The actual text of the token.
        """
        self.embedding = embedding  # Embedding vector (can be None for the root node)
        self.token_text = token_text
        self.children: Dict[int, 'Node'] = {}  # Mapping from token index to child Node
        self.token_indices: List[int] = []  # List of token indices where this node occurs
        self.text_mapping: Dict[Tuple[float, ...], str] = {}  # Embedding to token text
        self.index_to_text: Dict[int, str] = {}  # Mapping from token index to token text
        self.parent: Optional['Node'] = None  # Reference to the parent Node

        if embedding is not None:
            embedding_key = tuple(embedding)
            self.token_indices.append(token_index)
            self.text_mapping[embedding_key] = token_text
            self.index_to_text[token_index] = token_text
            logger.debug(f"Created Node: Token '{token_text}', Index {token_index}")
        else:
            # Handle the root node case where embedding is None
            logger.debug("Created Root Node")

    def add_child(self, token_index: int, child_node: 'Node'):
        """
        Add a child node to the current node.

        Args:
            token_index (int): The index of the token in the document.
            child_node (Node): The child node to add.
        """
        if token_index in self.children:
            logger.debug(f"Child at index {token_index} already exists. Overwriting.")
        self.children[token_index] = child_node
        child_node.parent = self
        logger.debug(f"Added child node at index {token_index} to node with indices {self.token_indices}")

    def update_embedding(self, new_embedding: np.ndarray):
        """
        Update the node's embedding.

        Args:
            new_embedding (np.ndarray): The new embedding vector.
        """
        self.embedding = new_embedding
        logger.debug("Node embedding updated.")

    def get_text(self, token_index: int) -> str:
        """
        Retrieve the text associated with the node's embedding at a specific token index.

        Args:
            token_index (int): The token index.

        Returns:
            str: The token text.
        """
        return self.index_to_text.get(token_index, "")

    def merge_with(self, other_node: 'Node'):
        """
        Merge another node into this node.

        Args:
            other_node (Node): The node to merge with.
        """
        # Merge token indices
        self.token_indices.extend(other_node.token_indices)
        # Merge text mappings
        self.text_mapping.update(other_node.text_mapping)
        self.index_to_text.update(other_node.index_to_text)
        # Merge children
        for idx, child in other_node.children.items():
            if idx not in self.children:
                self.children[idx] = child
                child.parent = self
            else:
                # If the child already exists, decide on a merging strategy
                self.children[idx].merge_with(child)
        logger.debug(f"Merged node with token indices {other_node.token_indices} into node with indices {self.token_indices}")

    def __repr__(self):
        return f"Node(token_indices={self.token_indices}, num_children={len(self.children)})"
    

#_____________________________________________________________________________________#
#_____________________________ HELPER FUNCTIONS ______________________________________#
#_____________________________________________________________________________________#
# Main function to construct the data structure
async def construct_embedding_graph(tokens: List[str], similarity_threshold: float) -> Tuple[Node, Dict[int, Node]]:
    """
    Construct the graph-like data structure from tokens.

    Args:
        tokens (List[str]): List of tokens from the document.
        similarity_threshold (float): Threshold for cosine similarity.

    Returns:
        Tuple[Node, Dict[int, Node]]: The root node and index-to-node mapping.
    """
    # Generate embeddings for all tokens
    embeddings = await get_embeddings(tokens)
    if embeddings.size == 0:
        raise ValueError("Embeddings could not be generated.")

    # Initialize root node (empty embedding)
    root = Node(embedding=None, token_index=-1, token_text="")
    prior_node = root

    # Keep a mapping from embeddings to nodes for quick similarity search
    embedding_to_node: Dict[Tuple[float, ...], Node] = {}
    index_to_node: Dict[int, Node] = {}

    for idx, (token, embedding) in enumerate(zip(tokens, embeddings)):
        logger.debug(f"Processing token {idx}: '{token}'")

        # Convert embedding to tuple for use as a dictionary key
        embedding_key = tuple(embedding)

        # Find the node with the closest embedding above the similarity threshold
        similar_node = None
        max_similarity = similarity_threshold  # Initialize with threshold

        for existing_embedding_key, existing_node in embedding_to_node.items():
            existing_embedding = np.array(existing_embedding_key)
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0][0]
            if similarity >= similarity_threshold and similarity > max_similarity:
                similar_node = existing_node
                max_similarity = similarity
                logger.debug(f"Found closer node for token '{token}' with similarity {similarity}")

        if similar_node:
            # Update the similar node
            similar_node.token_indices.append(idx)
            similar_node.text_mapping[embedding_key] = token
            similar_node.index_to_text[idx] = token
            current_node = similar_node
            logger.debug(f"Reusing node for token '{token}', indices now: {similar_node.token_indices}")
        else:
            # Create a new node
            current_node = Node(embedding=embedding, token_index=idx, token_text=token)
            embedding_to_node[embedding_key] = current_node
            logger.debug(f"Created new node for token '{token}' at index {idx}")

        index_to_node[idx] = current_node

        # Add the current node as a child of the prior node
        prior_node.add_child(idx, current_node)

        # Move to the next node
        prior_node = current_node

    logger.info("Completed constructing the embedding graph.")
    return root, index_to_node

# Function to reconstruct the document from the graph
def reconstruct_document(index_to_node: Dict[int, Node], num_tokens: int) -> str:
    """
    Reconstruct the document by iterating over the token indices.

    Args:
        index_to_node (Dict[int, Node]): Mapping from token index to node.
        num_tokens (int): Total number of tokens.

    Returns:
        str: The reconstructed document.
    """
    reconstructed_tokens = []
    for idx in range(num_tokens):
        node = index_to_node.get(idx)
        if node:
            token_text = node.index_to_text.get(idx, "")
            reconstructed_tokens.append(token_text)
            logger.debug(f"Token index {idx}: '{token_text}'")
        else:
            logger.warning(f"No node found for token index {idx}")
    return ''.join(reconstructed_tokens)