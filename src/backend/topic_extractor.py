from _util._utils import *
from .node import Node

class TopicExtractor:
    """
    Extracts topics from an embedding graph by creating separate graphs for each sliding window
    and identifying frequently occurring graph structures.
    """

    def __init__(self, tokens: List[str], index_to_node: Dict[int, Node], window_size: int):
        """
        Initialize the WindowedTopicExtractor.

        Args:
            tokens (List[str]): List of token texts.
            index_to_node (Dict[int, Node]): Mapping from token index to Node.
            window_size (int): Size of the sliding window.
        """
        self.tokens = tokens
        self.index_to_node = index_to_node
        self.window_size = window_size
        self.node_set_counter: Dict[str, int] = defaultdict(int)
        self.hash_to_node_set: Dict[str, tuple] = {}
        
    def identify_topics(self) -> List[tuple]:
        """
        Identify topics based on the frequency of graph structures across sliding windows.

        Args:
            min_occurrence (int): Minimum number of occurrences for a graph to be considered a topic.

        Returns:
            List[nx.Graph]: List of topic graphs.
        """
        self._process_windows()
        topics = [(self.hash_to_node_set[node_set_hash], count) 
                  for node_set_hash, count in self.node_set_counter.items()]
        
        topics = self._filter_frequent_topics(topics)
        topics = self._merge_similar_topics(topics) # merge similar topic

        return topics

    def _process_windows(self):
        """
        Process each sliding window, create its graph, hash it, and count its occurrences.
        """
        num_tokens = len(self.tokens)

        for i in tqdm(range(0, num_tokens - self.window_size + 1, 2), desc="Processing windows"): # step by 2 to step over whitespace nodes.
            window_indices = list(range(i, i + self.window_size))
            window_nodes = self._create_window_nodes(window_indices)
            node_set_hash = self._hash_node_set(window_nodes)

            self.node_set_counter[node_set_hash] += 1
            if node_set_hash not in self.hash_to_node_set:
                self.hash_to_node_set[node_set_hash] = (window_nodes, window_indices)

    def _create_window_nodes(self, window_indices: range) -> set:
        """
        Create a graph for a given sliding window.

        Args:
            window_indices (range): Range of token indices in the window.

        Returns:
            nx.Graph: Graph representing the co-occurrence of nodes within the window.
        """
        # graph = nx.Graph()
        ordered_window_nodes = set([])
        window_nodes = []
        num_whitespace_nodes = 0
        for idx in window_indices:
            node = self.index_to_node[idx]
            if not node.is_ignorable_node:
                window_nodes.append(node)
            else:
                num_whitespace_nodes += 1

        for node in window_nodes:
            ordered_window_nodes = ordered_window_nodes.union([self._serialize_node(node)])

        return ordered_window_nodes

    def _serialize_node(self, node: Node) -> str:
        """
        Serialize a node to a string representation.

        Args:
            node (Node): The node to serialize.

        Returns:
            str: Serialized node identifier.
        """
        return node

    def _hash_node_set(self, node_set: set) -> str:
        """
        Generate a hash for a given graph structure.

        Args:
            graph (nx.Graph): The graph to hash.

        Returns:
            str: The SHA256 hash of the graph.
        """
        graph_data = {
            "nodes": sorted([x.token_text for x in node_set]),
        }
        graph_json = json.dumps(graph_data).encode('utf-8')
        return graph_json # hashlib.sha256(graph_json).hexdigest() # no need for the hashing complexity
    

    def map_topics_to_tokens(self, topics: List[tuple], index_to_node: Dict[int, Node]) -> List[List[str]]:
        """
        Map each topic graph's nodes back to their associated token texts.

        Args:
            topics (List[nx.Graph]): List of topic graphs.
            hash_to_node_set (Dict[str, nx.Graph]): Mapping from graph hashes to graph objects.

        Returns:
            List[List[str]]: List of topics, each being a list of token texts.
        """

        topic_tokens = []

        for topic_graph_w_indices, count in topics:
            topic_nodes, window_indices = topic_graph_w_indices

            aggregate_maps = defaultdict(str)

            for node in topic_nodes:
                aggregate_maps.update(node.index_to_text)

        
            tokens = []
            for index in window_indices:
                
                if USE_ACTUAL_TEXT:
                    temp_token = aggregate_maps[index].strip()
                else:
                    temp_token = index_to_node[index].token_text.strip()
                
                # deduplicate the token space
                if temp_token not in tokens:
                    tokens.append(temp_token)
                
            topic_tokens.append((tokens, count))

        # sort based on topic frequency
        topic_tokens.sort(key = lambda x: x[1])
        
        # keep only topics that meet the `TOPIC_MIN_OCCURRENCE` threshold
        topic_tokens = [[f'({x[1]})'] + x[0] for x in topic_tokens]
        
        return topic_tokens

    def _filter_frequent_topics(self, topics: List[tuple]) -> List[tuple]:
        """
        Filter out frequent tokens from the list of tokens.

        Args:
            tokens (List[str]): List of token texts.

        Returns:
            List[str]: Filtered list of token texts.
        """
        total_topics = len(topics)
        node_counts = {}

        # Count occurrences of each token across all topics
        for topic, _ in topics:
            for node in topic[0]:
                node_counts[node] = node_counts.get(node, 0) + 1

        # Identify redundant tokens
        redundant_nodes = {node for node, count in node_counts.items() if count / total_topics >= REDUNDANT_TOKEN_THRESHOLD}

        logging.info(f"Filtering {len(redundant_nodes)} redundant nodes")
        
        for topic, _ in topics:
            for node in list(topic[0]):
                if node in redundant_nodes:
                    topic[0].remove(node)

        return topics

    def _merge_similar_topics(self, topics: List[Tuple[nx.Graph, int]]) -> List[Tuple[nx.Graph, int]]:
        """
        Merge topics that have significant overlap and remove redundant tokens.

        Args:
            topics (List[Tuple[nx.Graph, int]]): List of topics and their frequencies.

        Returns:
            List[Tuple[nx.Graph, int]]: Merged list of topics and their combined frequencies.
        """
        # Precompute node sets for all topics
        preprocessed_topics = [
            {
                'nodes': topic_nodes,
                'range': topic_range,
                'count': count
            }
            for (topic_nodes, topic_range), count in topics
        ]

        # Sort topics by frequency in descending order to prioritize merging with larger clusters
        preprocessed_topics.sort(key=lambda x: x['count'])

        clusters = []

        for topic in tqdm(preprocessed_topics, desc="Merging topics"):
            merged = False
            topic_nodes = topic['nodes']
            topic_range = topic['range']
            count = topic['count']

            for cluster in clusters:
                cluster_nodes = cluster['nodes']
                # Compute Jaccard similarity based on precomputed node sets
                intersection_size = len(topic_nodes & cluster_nodes)
                union_size = len(topic_nodes | cluster_nodes)
                similarity = intersection_size / union_size if union_size else 0.0

                if similarity >= MERGE_SIMILAR_TOPICS_THRESHOLD:
                    cluster['range'] += topic_range
                    cluster['count'] += count
                    # Update the node set
                    cluster['nodes'].update(topic_nodes)
                    merged = True
                    break  # Assuming each topic can only belong to one cluster

            if not merged:
                # If no similar cluster found, add as a new cluster with cached node set
                clusters.append({
                    'nodes': topic_nodes,  # Ensure a separate copy
                    'range': topic_range,
                    'count': count
                })

        # Sort clusters by frequency in descending order
        clusters.sort(key=lambda x: x['count'])

        # Convert clusters back to the desired return format
        merged_clusters = [
            ((cluster['nodes'], cluster['range']), cluster['count']) for cluster in clusters
        ]

        return merged_clusters

    def _get_node_by_id(self, node_id: str) -> Node:
        """
        Retrieve a Node instance based on its serialized identifier.

        Args:
            node_id (str): Serialized node identifier.

        Returns:
            Node: The corresponding Node instance or None if not found.
        """
        # Extract the unique identifier from the serialized node_id
        unique_id = node_id.split('Node_')[1]
        for node in self.index_to_node.values():
            if node.token_text == unique_id:
                return node
        return None