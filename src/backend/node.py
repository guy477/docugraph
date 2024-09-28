from _util._utils import *

import faiss

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

        # Attributes to track whitespace and numeric ratios
        self._total_tokens: int = 0
        self._whitespace_tokens: int = 0
        self._numeric_tokens: int = 0
        self.is_ignorable_node: bool = False  # Initialized based on ratios

        if embedding is not None:
            embedding_key = tuple(embedding)
            self.add_token(token_index, token_text, embedding_key)
            logger.debug(f"Created Node: Token '{token_text}', Index {token_index}")
        else:
            # Handle the root node case where embedding is None
            logger.debug("Created Root Node")

    def add_token(self, token_index: int, token_text: str, embedding_key: Tuple[float, ...]):
        """
        Add a token to the node and update whitespace and numeric tracking.

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
        if any(char.isdigit() for char in token_text):
            self._numeric_tokens += 1
        self._update_ignorable_flag()
        logger.debug(f"Added token '{token_text}' at index {token_index}. "
                        f"Whitespace tokens: {self._whitespace_tokens}/{self._total_tokens}. "
                        f"Numeric tokens: {self._numeric_tokens}/{self._total_tokens}. "
                        f"Is ignorable node: {self.is_ignorable_node}")

    def remove_token(self, token_index: int, embedding_key: Tuple[float, ...]):
        """
        Remove a token from the node.

        Args:
            token_index (int): The index of the token to remove.
            embedding_key (Tuple[float, ...]): The embedding key of the token.
        """
        if token_index in self.token_indices:
            self.token_indices.remove(token_index)
            del self.index_to_text[token_index]
            del self.text_mapping[embedding_key]
            self._total_tokens -= 1
            # Assuming token_text is available
            token_text = self.index_to_text.get(token_index, "")
            if token_text.isspace():
                self._whitespace_tokens -= 1
            if any(char.isdigit() for char in token_text):
                self._numeric_tokens -= 1
            self._update_ignorable_flag()
            logger.debug(f"Removed token '{token_text}' from Node {self}.")
        else:
            logger.warning(f"Token index {token_index} not found in Node {self}.")

    def _update_ignorable_flag(self):
        """
        Update the is_ignorable_node flag based on the current whitespace and numeric ratios.
        """
        if self._total_tokens == 0:
            self.is_ignorable_node = False
        else:
            whitespace_ratio = self._whitespace_tokens / self._total_tokens
            numeric_ratio = self._numeric_tokens / self._total_tokens
            self.is_ignorable_node = whitespace_ratio >= WHITESPACE_THRESHOLD or numeric_ratio >= WHITESPACE_THRESHOLD

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
        # Merge whitespace and numeric tracking
        self._total_tokens += other_node._total_tokens
        self._whitespace_tokens += other_node._whitespace_tokens
        self._numeric_tokens += other_node._numeric_tokens
        self._update_ignorable_flag()
        # Merge children
        for idx, child in other_node.children.items():
            if idx not in self.children:
                self.children[idx] = child
            else:
                # If the child already exists, decide on a merging strategy
                self.children[idx].merge_with(child)
        logger.debug(f"Merged node with token indices {other_node.token_indices} into node with indices {self.token_indices}. "
                     f"Is ignorable node: {self.is_ignorable_node}")

    def __repr__(self):
        return (f"Node(token_indices={self.token_indices}, num_children={len(self.children)}, "
                f"is_ignorable={self.is_ignorable_node})")


class Rebalancer:
    """
    Handles the rebalancing of the embedding graph by migrating tokens
    to more appropriate nodes based on embedding similarity.
    """

    def __init__(self, root: Node, index_to_node: Dict[int, Node]):
        """
        Initialize the Rebalancer.

        Args:
            root (Node): The root node of the embedding graph.
            index_to_node (Dict[int, Node]): Mapping from token index to node.
        """
        self.root = root
        self.index_to_node = index_to_node
        self.embedding_cache: Dict[Tuple[float, ...], Node] = {}

    def rebalance_graph(self):
        """
        Rebalance the entire embedding graph by reassigning tokens to the most suitable nodes.
        """
        logger.info("Starting graph rebalancing.")
        num_migrations = 0
        tokens = self._gather_all_tokens()
        for token_index, token_text, embedding in tqdm(tokens, desc="Rebalancing graph"):
            num_migrations += self._process_token(token_index, token_text, embedding)

        logger.info(f"Completed graph rebalancing with {num_migrations} tokens migrated.")

    def _gather_all_tokens(self) -> List[Tuple[int, str, Tuple[float, ...]]]:
        """
        Gather all tokens from all nodes in the graph.

        Returns:
            List[Tuple[int, str, Tuple[float, ...]]]: A list of tuples containing token index, text, and embedding.
        """
        tokens = []
        for node in set(self.index_to_node.values()): # order shouldn't matter
            for idx in node.token_indices:
                token_text = node.get_text(idx)
                embedding = node.text_mapping.get(tuple(node.embedding), "")
                if embedding:
                    tokens.append((idx, token_text, tuple(node.embedding)))
        logger.debug(f"Gathered {len(tokens)} tokens for rebalancing.")
        return tokens

    def _process_token(self, token_index: int, token_text: str, embedding: Tuple[float, ...]):
        """
        Process a single token by determining if it should be moved to a better node.

        Args:
            token_index (int): The index of the token.
            token_text (str): The text of the token.
            embedding (Tuple[float, ...]): The embedding vector of the token.
        """
        current_node = self.index_to_node.get(token_index)
        if not current_node:
            logger.warning(f"No current node found for token index {token_index}. Skipping.")
            return

        best_node = self._find_best_node(embedding)
        total_migrations = 0
        if best_node and best_node != current_node:
            logger.info(f"Token '{token_text}' at index {token_index} will be moved from Node {current_node} to Node {best_node}.")
            self._migrate_token(token_index, token_text, embedding, current_node, best_node)
            total_migrations += 1
        else:
            logger.debug(f"Token '{token_text}' at index {token_index} remains in Node {current_node}.")

        return total_migrations
    def _find_best_node(self, embedding: Tuple[float, ...]) -> Optional[Node]:
        """
        Find the best node for a given embedding based on the similarity threshold.

        Args:
            embedding (Tuple[float, ...]): The embedding vector to find a node for.

        Returns:
            Optional[Node]: The best matching node or None if no suitable node is found.
        """
        if embedding in self.embedding_cache:
            logger.debug("Cache hit for embedding.")
            return self.embedding_cache[embedding]

        best_similarity = -1.0
        best_node = None
        for node in self.index_to_node.values():
            if node.embedding is None:
                continue  # Skip root node
            similarity = self._compute_similarity(embedding, tuple(node.embedding))
            if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_node = node

        if best_node:
            self.embedding_cache[embedding] = best_node
            logger.debug(f"Best node found with similarity {best_similarity:.4f}.")
        else:
            logger.debug("No suitable best node found.")

        return best_node

    def _compute_similarity(self, emb1: Tuple[float, ...], emb2: Tuple[float, ...]) -> float:
        """
        Compute the cosine similarity between two embeddings.

        Args:
            emb1 (Tuple[float, ...]): The first embedding vector.
            emb2 (Tuple[float, ...]): The second embedding vector.

        Returns:
            float: The cosine similarity.
        """
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity

    def _migrate_token(self, token_index: int, token_text: str, embedding: Tuple[float, ...],
                       current_node: Node, best_node: Node):
        """
        Migrate a token from the current node to the best node.

        Args:
            token_index (int): The index of the token.
            token_text (str): The text of the token.
            embedding (Tuple[float, ...]): The embedding vector of the token.
            current_node (Node): The node from which the token is being migrated.
            best_node (Node): The node to which the token is being migrated.
        """
        # Remove token from current node
        current_node.token_indices.remove(token_index)
        del current_node.index_to_text[token_index]
        del current_node.text_mapping[embedding]
        current_node._total_tokens -= 1
        if token_text.isspace():
            current_node._whitespace_tokens -= 1
        current_node._update_whitespace_flag()

        # Add token to best node
        best_node.add_token(token_index, token_text, embedding)

        # Update the index_to_node mapping
        self.index_to_node[token_index] = best_node

        logger.debug(f"Migrated token index {token_index} to Node {best_node}.")

#_____________________________________________________________________________________#
#_____________________________ HELPER FUNCTIONS ______________________________________#
#_____________________________________________________________________________________#

async def build_and_rebalance_graph(tokens: List[str]) -> Tuple[Node, Dict[int, Node]]:
    """
    Constructs the embedding graph and then rebalances it.

    Args:
        tokens (List[str]): List of tokens from the document.

    Returns:
        Tuple[Node, Dict[int, Node]]: The root node and index-to-node mapping.
    """
    root, index_to_node = await construct_embedding_graph(tokens)
    
    # Initialize the Rebalancer
    rebalancer = Rebalancer(root, index_to_node)
    
    # Perform rebalancing
    # NOTE: From experimentation, rebalancing is a no-op.
    #  This suggests every token is assigned to the best node.
    #  After construction. It's also really slow. Leave commented.
    # rebalancer.rebalance_graph()
    
    return root, index_to_node


async def construct_embedding_graph(tokens: List[str]) -> Tuple[Node, Dict[int, Node]]:
    """
    Optimized construction of the embedding graph using FAISS for fast similarity search.
    
    Args:
        tokens (List[str]): List of tokens from the document.
    
    Returns:
        Tuple[Node, Dict[int, Node]]: The root node and index-to-node mapping.
    """
    # Generate embeddings for all tokens
    embeddings = await get_embeddings(tokens)
    if embeddings.size == 0:
        raise ValueError("Embeddings could not be generated.")
    
    # Normalize all embeddings to unit vectors to simplify cosine similarity to dot product
    normalized_embeddings = normalize(embeddings, axis=1).astype('float32')  # FAISS requires float32
    
    # Initialize FAISS index
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    
    # Initialize root node (empty embedding)
    root = Node(embedding=None, token_index=-1, token_text="")
    prior_node = root
    
    # Mapping from token index to node
    index_to_node: Dict[int, Node] = {}
    node_list: List[Node] = []
    
    for idx in tqdm(range(len(tokens)), desc="Processing tokens"):
        token = tokens[idx]
        norm_emb = normalized_embeddings[idx]
        
        logger.debug(f"Processing token {idx}: '{token}'")
        
        if index.ntotal > 0:
            # Perform a search for the most similar node
            # Since we need the most similar node with similarity >= threshold,
            # we search for top 1 and check the similarity
            D, I = index.search(norm_emb.reshape(1, -1), 1)
            max_similarity = D[0][0]
            max_idx = I[0][0]
            
            if max_similarity >= SIMILARITY_THRESHOLD:
                similar_node = node_list[max_idx]
                logger.debug(f"Found similar node for token '{token}' with similarity {max_similarity:.4f}")
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
            node_list.append(current_node)
            # Add the normalized embedding to FAISS index
            index.add(np.expand_dims(norm_emb, axis=0))  # FAISS expects 2D array
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