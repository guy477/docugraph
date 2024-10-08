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
        """
        previous_non_whitespace_id = None

        for idx in range(self.num_tokens):
            node = self.index_to_nodes.get(idx)
            if not node:
                logger.warning(f"No node found for token index {idx}")
                continue

            if node.is_ignorable_node:
                # Skip whitespace nodes
                continue

            current_id = self._get_or_assign_node_id(node)
            self.graph.add_node(current_id, label=self._get_node_label(node))

            if previous_non_whitespace_id is not None:
                self._accumulate_edge(previous_non_whitespace_id, current_id)

            previous_non_whitespace_id = current_id

        self._bulk_add_edges()
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

    def _accumulate_edge(self, from_id: int, to_id: int):
        """
        Accumulate the edge weight between two nodes without modifying the graph directly.

        Args:
            from_id (int): The ID of the source node.
            to_id (int): The ID of the target node.
        """
        edge = (from_id, to_id)
        if edge in self.edge_weights:
            self.edge_weights[edge] += 1
        else:
            self.edge_weights[edge] = 1
        logger.debug(
            f"Accumulated edge between {from_id} and {to_id} with current weight {self.edge_weights[edge]}"
        )

    def _bulk_add_edges(self):
        """
        Add all accumulated edges to the graph in bulk with their corresponding weights.
        """
        edges_to_add = [
            (from_id, to_id, {'weight': weight})
            for (from_id, to_id), weight in self.edge_weights.items()
        ]
        self.graph.add_edges_from(edges_to_add)
        logger.debug(f"Bulk added {len(edges_to_add)} edges to the graph.")

    def visualize(self, output_path: str):
        """
        Visualize the graph using matplotlib and save it to a file.

        Args:
            output_path (str): The file path to save the visualization.
        """
        self.build_graph()
        logger.info(f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        pos = graphviz_layout(self.graph, prog='twopi')
        labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        node_colors = ['lightblue' for _ in self.graph.nodes()]

        plt.figure(figsize=(24, 24))

        logger.info(f"Drawing graph....")
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_size=1000,
            node_color=node_colors,
            edge_color='lightgray',
            arrows=True,
            linewidths=1.5,
            font_size=10
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
        if all(text.isspace() for text in token_texts):
            tokens_display = 'WHITESPACE'
        else:
            tokens_display = ', '.join(token_texts[:3])
            tokens_display += '...' if len(token_texts) > 3 else ''



        label = f"{node.token_text}:\n{'{'}{tokens_display}{'}'}"
        return label
