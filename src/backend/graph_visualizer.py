from _util._utils import *
from .node import Node

class GraphVisualizer:
    def __init__(self, index_to_node: Dict[int, Node], num_tokens: int):
        """
        Initialize the GraphVisualizer with the root node of the graph.

        Args:
            index_to_node (Dict[int, Node]): Mapping from token index to node.
            num_tokens (int): Total number of tokens.
        """
        self.index_to_nodes = index_to_node
        self.num_tokens = num_tokens
        self.graph = nx.DiGraph()  # Directed graph
        self.node_counter = 0  # Counter to assign unique IDs to node instances
        self.node_to_id: Dict[Node, int] = {}  # Maps Node instances to unique IDs
        self.edge_weights: Dict[Tuple[int, int], int] = {}  # Directed edges
        self.visited_pairs: Set[Tuple[int, int]] = set()  # Tracks visited (parent_id, current_id) pairs
        self.visited: Set[int] = set()  # Tracks visited node IDs

    def build_graph(self):
        """
        Build the graph representation using chronological traversal based on index_to_node,
        while avoiding whitespace nodes by maintaining the last non-whitespace node.

        Args:
            index_to_node (Dict[int, Node]): Mapping from token index to node.
            num_tokens (int): Total number of tokens.
        """
        previous_non_whitespace_id = None

        for idx in range(self.num_tokens):
            node = self.index_to_nodes.get(idx)
            if not node:
                logger.warning(f"No node found for token index {idx}")
                continue

            if self._is_whitespace_node(node):
                # Skip whitespace nodes
                continue

            current_id = self._get_or_assign_node_id(node)
            self.graph.add_node(current_id, label=self._get_node_label(node))

            if previous_non_whitespace_id is not None:
                self._add_or_update_edge(previous_non_whitespace_id, current_id)

            previous_non_whitespace_id = current_id

        logger.info("Completed building the graph from index_to_node while avoiding whitespace nodes.")

    def _get_or_assign_node_id(self, node: Node) -> int:
        """
        Retrieve the unique ID for a node, assigning a new one if it doesn't exist.

        Args:
            node (Node): The node to retrieve or assign an ID for.

        Returns:
            int: The unique ID of the node.
        """
        if node not in self.node_to_id:
            self.node_to_id[node] = self.node_counter
            self.node_counter += 1
        return self.node_to_id[node]

    def _add_or_update_edge(self, from_id: int, to_id: int):
        """
        Add a new edge or update the weight of an existing edge between two nodes.

        Args:
            from_id (int): The ID of the source node.
            to_id (int): The ID of the target node.
        """
        edge = (from_id, to_id)
        if edge in self.edge_weights:
            self.edge_weights[edge] += 1
        else:
            self.edge_weights[edge] = 1
        self.graph.add_edge(from_id, to_id, weight=self.edge_weights[edge])
        logger.debug(
            f"Added/Updated edge between {from_id} and {to_id} with weight {self.edge_weights[edge]}"
        )

    def visualize(self, output_path: str):
        """
        Visualize the graph using matplotlib and save it to a file.

        Args:
            output_path (str): The file path to save the visualization.
        """
        self.build_graph()
        pos = graphviz_layout(self.graph, prog='circo')
        labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        node_colors = ['lightblue' for _ in self.graph.nodes()]

        plt.figure(figsize=(18, 18))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_size=500,
            node_color=node_colors,
            edge_color='lightgray',
            arrows=True,
            linewidths=1.5,
            font_size=12
        )
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Graph visualization saved to {output_path}")

    def _get_node_label(self, node: Node) -> str:
        """
        Generate a label for a node, including token texts.

        Args:
            node (Node): The node for which to generate the label.

        Returns:
            str: A string label for the node.
        """
        token_texts = list(set([node.index_to_text.get(idx, '') for idx in node.token_indices]))
        if all(re.match(r'\s+', text) for text in token_texts):
            tokens_display = 'WHITESPACE'
        else:
            tokens_display = ', '.join(token_texts[:3])
            tokens_display += '...' if len(token_texts) > 3 else ''



        label = f"{node.token_text}:\n{'{'}{tokens_display}{'}'}"
        return label

    def _is_whitespace_node(self, node: Node) -> bool:
        """
        Determine if a node is a whitespace node based on a threshold percentage.

        Args:
            node (Node): The node to check.
            threshold (float, optional): The percentage of whitespace required to return True. Default is 90%.

        Returns:
            bool: True if the percentage of whitespace exceeds the threshold, False otherwise.
        """
        matches = [bool(re.match(r'^\s+$', text)) for text in node.index_to_text.values()]
        true_count = sum(matches)
        total_count = len(matches)

        if total_count == 0:
            return False

        whitespace_prob = true_count / total_count
        logger.debug(f"Whitespace probability for node {node}: {whitespace_prob}")

        return whitespace_prob >= WHITESPACE_THRESHOLD