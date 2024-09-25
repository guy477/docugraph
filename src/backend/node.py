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

        # Attributes to track whitespace ratio
        self._total_tokens: int = 0
        self._whitespace_tokens: int = 0
        self.is_whitespace_node: bool = False  # Initialized based on ratio

        if embedding is not None:
            embedding_key = tuple(embedding)
            self.add_token(token_index, token_text, embedding_key)
            logger.debug(f"Created Node: Token '{token_text}', Index {token_index}")
        else:
            # Handle the root node case where embedding is None
            logger.debug("Created Root Node")

    def add_token(self, token_index: int, token_text: str, embedding_key: Tuple[float, ...]):
        """
        Add a token to the node and update whitespace tracking.

        Args:
            token_index (int): The index of the token in the document.
            token_text (str): The text of the token.
            embedding_key (Tuple[float, ...]): The embedding key for the token.
        """
        self.token_indices.append(token_index)
        self.text_mapping[embedding_key] = token_text
        self.index_to_text[token_index] = token_text
        self._total_tokens += 1
        if token_text.isspace():
            self._whitespace_tokens += 1
        self._update_whitespace_flag()
        logger.debug(f"Added token '{token_text}' at index {token_index}. "
                     f"Whitelist tokens: {self._whitespace_tokens}/{self._total_tokens}. "
                     f"Is whitespace node: {self.is_whitespace_node}")

    def _update_whitespace_flag(self):
        """
        Update the is_whitespace_node flag based on the current whitespace ratio.
        """
        if self._total_tokens == 0:
            self.is_whitespace_node = False
        else:
            whitespace_ratio = self._whitespace_tokens / self._total_tokens
            self.is_whitespace_node = whitespace_ratio >= WHITESPACE_THRESHOLD

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
        (UNTESTED) Merge another node into this node. 

        Args:
            other_node (Node): The node to merge with.
        """
        # Merge token indices
        self.token_indices.extend(other_node.token_indices)
        # Merge text mappings
        self.text_mapping.update(other_node.text_mapping)
        self.index_to_text.update(other_node.index_to_text)
        # Merge whitespace tracking
        self._total_tokens += other_node._total_tokens
        self._whitespace_tokens += other_node._whitespace_tokens
        self._update_whitespace_flag()
        # Merge children
        for idx, child in other_node.children.items():
            if idx not in self.children:
                self.children[idx] = child
                child.parent = self
            else:
                # If the child already exists, decide on a merging strategy
                self.children[idx].merge_with(child)
        logger.debug(f"Merged node with token indices {other_node.token_indices} into node with indices {self.token_indices}. "
                     f"Is whitespace node: {self.is_whitespace_node}")

    def __repr__(self):
        return (f"Node(token_indices={self.token_indices}, num_children={len(self.children)}, "
                f"is_whitespace={self.is_whitespace_node})")


#_____________________________________________________________________________________#
#_____________________________ HELPER FUNCTIONS ______________________________________#
#_____________________________________________________________________________________#


async def construct_embedding_graph(tokens: List[str], similarity_threshold: float) -> Tuple[Node, Dict[int, Node]]:
    """
    Optimized construction of the embedding graph using vectorized cosine similarity.
    
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
    
    # Normalize all embeddings to unit vectors to simplify cosine similarity to dot product
    normalized_embeddings = normalize(embeddings, axis=1)
    
    # Initialize root node (empty embedding)
    root = Node(embedding=None, token_index=-1, token_text="")
    prior_node = root
    
    # Lists to store normalized embeddings and corresponding nodes
    embedding_list = []  # List of normalized embeddings (numpy arrays)
    node_list = []       # List of Node objects corresponding to embeddings_list
    
    # Mapping from token index to node
    index_to_node: Dict[int, Node] = {}
    
    for idx, (token, norm_emb) in tqdm(enumerate(zip(tokens, normalized_embeddings)), desc="Processing tokens", total=len(tokens)):
        logger.debug(f"Processing token {idx}: '{token}'")
        
        if embedding_list:
            # Stack existing normalized embeddings into a 2D array for vectorized operations
            existing_embeddings = np.vstack(embedding_list)  # Shape: (N, D)
            
            # Compute cosine similarities using dot product since embeddings are normalized
            similarities = existing_embeddings @ norm_emb  # Shape: (N,)
            
            # Find indices where similarity meets or exceeds the threshold
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            
            if valid_indices.size > 0:
                # Select the index with the highest similarity
                max_sim_idx = valid_indices[np.argmax(similarities[valid_indices])]
                similar_node = node_list[max_sim_idx]
                logger.debug(f"Found similar node for token '{token}' with similarity {similarities[max_sim_idx]:.4f}")
            else:
                similar_node = None
        else:
            similar_node = None
        
        if similar_node:
            # Reuse the similar node
            similar_node.add_token(idx, token, tuple(embeddings[idx]))
            current_node = similar_node
            logger.debug(f"Reusing node for token '{token}', indices now: {similar_node.token_indices}")
        else:
            # Create a new node
            current_node = Node(embedding=embeddings[idx], token_index=idx, token_text=token)
            embedding_list.append(norm_emb)  # Store normalized embedding
            node_list.append(current_node)
            logger.debug(f"Created new node for token '{token}' at index {idx}")
        
        # Map the current token index to the current node
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
            token_text = node.get_text(idx)
            reconstructed_tokens.append(token_text)
            logger.debug(f"Token index {idx}: '{token_text}'")
        else:
            logger.warning(f"No node found for token index {idx}")
    return ''.join(reconstructed_tokens)