from __future__ import print_function
import numpy as np
from collections import OrderedDict, deque, defaultdict
import hashlib


class ItemRanker(object):
    """
    ItemRanker: A ranking class that uses graph coloring and round-robin algorithms to rank items based on their categorical attributes.
    The goal of this class is to provide balanced placement of item categories within the ranked list.

     Features:
    ---------
    - Graph coloring: Assigns items to categories ensuring no two adjacent items are of the same category.
    - Round-robin ranking: Provides a fallback mechanism for ranking when graph coloring fails.
    - Caching: Efficiently stores and retrieves computed rankings for reuse.

    Attributes:
    ----------
    cache : dict
        A class-level dictionary for caching rankings.

    """

    cache = {}

    def __init__(self, candidateItems, item_dataframe, dimension, **kwargs):
        """
        Initialize the ItemRanker class with candidate items, item feature dataframe, and a diversity dimension.

        Parameters:
        -----------
            candidateItems : list
            List of item indices to be ranked.

            item_dataframe : pd.DataFrame
                DataFrame containing item attributes such as categories, features, or dimensions.

            dimension : str
                The column name in the DataFrame representing the attribute (e.g., category) to be used for ranking.

        Attributes:
        -----------
            V : int
                The number of candidate items to be ranked.

            dimension : str
                The diversity dimension used for ranking.

            candidateItems : list
                The list of item indices provided for ranking.

            item_dataframe : pd.Series
                A filtered view of the item DataFrame containing the values of the specified dimension for the candidate items.

            color_dict : OrderedDict
                An ordered dictionary mapping each category (in the specified dimension) to its count of items among the candidates.

            used_color : OrderedDict
                Tracks the number of times each category (color) has been assigned during the graph coloring process, initialized to zero.

            exceeded_max_depth : bool
                A flag to indicate if the maximum recursion depth was exceeded during graph coloring.

        """
        self._validate_input(candidateItems, item_dataframe, dimension)
        self.V = len(candidateItems)
        self.dimension = dimension

        self.candidateItems = candidateItems
        # Create a DataFrame view for the selected candidate items based on the given dimension
        self.item_dataframe = item_dataframe.loc[self.candidateItems, dimension]
        # Count the occurrences of each category in the selected dimension
        category_counts = self.item_dataframe.value_counts()

        self.color_dict = OrderedDict(
            category_counts.to_dict())

        self.used_color = OrderedDict((key, 0)
                                      for key in self.color_dict.keys())
        self.exceeded_max_depth = False  # Initialize the flag for depth limit

         # Handle optional parameters in kwargs
        self.random_walk_score = kwargs.get("random_walk_score", None)
        if self.random_walk_score is not None:
            # print(f"Random Walk Score Loaded:{self.random_walk_score}")
            print("Random Walk Score Loaded")

    def _validate_input(self, candidateItems, articleDataframe, dimension):
        """
        Validate input parameters to ensure proper initialization.

        Raises:
        -------
        TypeError : If candidateItems is not a list or dimension is not a string.
        ValueError : If the dimension is not found in the DataFrame columns.
        IndexError : If candidate items are not a subset of DataFrame indices.
        """
        if not isinstance(candidateItems, list):
            raise TypeError(
                f"candidateItems should be a list, but got {type(candidateItems)}.")

        if not isinstance(dimension, str):
            raise TypeError(
                f"dimension should be a string, but got {type(dimension)}.")

        if dimension not in articleDataframe.columns:
            raise ValueError(
                f"Dimension '{dimension}' not found in the DataFrame columns.")

        if not set(candidateItems).issubset(articleDataframe.index):
            invalid_items = set(candidateItems) - set(articleDataframe.index)
            raise IndexError(
                f"The following candidateItems indices are invalid: {invalid_items}.")

    def _generate_cache_key(self):
        """
        Generate a unique cache key based on the candidate items and the dimension.

        Returns:
        --------
        str : A hashed string representing the cache key.
        """
        key_string = f"{self.candidateItems}-{self.V}-{self.dimension}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def create_color_sequence(self, color):
        """
        Generate the final color assignment (category order) for the items.

        Parameters:
        color (list): List of color indices representing category assignments for each item.

        Returns:
        list: List of actual category labels.
        """
        # Convert color indices to actual category labels .
        result_coloring = list(self.used_color.keys())
        result_coloring = [result_coloring[i] for i in color]
        return (result_coloring)

    def buildAdjMatrix(self):
        """
        Build the adjacency matrix for the graph where each item is connected to its neighbors.

        Returns:
        list: A 2D list representing the adjacency matrix of the items.
        """
        N = self.V
        graph = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N-1):  # V-1 because the last row does not have a column to its right
            graph[i][i+1] = 1  # Connect current item to next
            graph[i+1][i] = 1  # Connect next item back to current
        return graph

    def is_valid_color(self, v, graph, color, c):
        """ Check if the color 'c' is valid for the vertex 'v'.

            The function checks two conditions:
            1. The selected color (category) 'c' has not been used more times than allowed by its count in the dataset.
            2. No adjacent vertex to 'v' (as defined by the adjacency matrix) has already been assigned the same color 'c'.

            Parameters:
            v (int): The vertex index (item)for assigning a color.
            graph (list of lists): The adjacency matrix representing the connections between items.
            color (list): The current color assignment of all vertices (items).
            c (int): The color index to check for validity.

            Returns:
            bool: True if the color 'c' can be assigned to vertex 'v', False otherwise.

        """
        # Get the actual color (category) associated with color index 'c'
        original_color = list(self.used_color.keys())[c]
        # Check if the color 'original_color' has been fully used
        if self.used_color[original_color] >= self.color_dict[original_color]:
            return False
         # Check if any adjacent vertex to 'v' has already been assigned the same color 'c'
        for i in range(self.V):
            if graph[v][i] and c == color[i]:
                return False
        return True

    def graph_coloring(self, graph, m, color, v, recursion_depth=0, max_depth=15):
        """
        Recursively solve the graph coloring problem using backtracking.

        Parameters:
        graph (list): The adjacency matrix of the items.
        m (int): Number of available colors (categories).
        color (list): List of current color assignments for all vertices.
        v (int): The current vertex to color.

        Returns:
        bool: True if a valid coloring is found, False otherwise.
        """
        if self.exceeded_max_depth:
            return False
        if recursion_depth > max_depth:
            self.exceeded_max_depth = True
            return False
        # Base case: If all vertices are assigned a color, return true
        if v == self.V:
            return True

        # Try different colors for the current vertex v
        for c in range(0, m):
            # Check if assignment of color 'c' to v is ok
            if self.is_valid_color(v, graph, color, c):
                color[v] = c
                original_color = list(self.used_color.keys())[c]
                self.used_color[original_color] += 1
                # Recur to assign colors to the rest of the vertices
                if self.graph_coloring(graph, m, color, v + 1, recursion_depth + 1, max_depth):
                    return True

                # If assigning color 'c' doesn't lead to a solution, remove it
                color[v] = -1
                self.used_color[original_color] -= 1
        # If no color can be assigned to this vertex, return false
        return False

    def solve_graph_coloring(self):
        """
        Solve the graph coloring problem and return the ordered categories.

        Returns:
        list: List of categories in the order of the final color assignment, or an empty list if no solution.
        """
       # Reset the exceeded depth flag
        self.exceeded_max_depth = False
        graph = self.buildAdjMatrix()
        color = [-1] * self.V
        m = len(self.color_dict.keys())
        if not self.graph_coloring(graph, m, color, 0):
            return []
        result = self.create_color_sequence(color)
        # reset used_color
        self.used_color = OrderedDict((key, 0)
                                      for key in self.color_dict.keys())
        return (result)

    def round_robin_rank(self):
        """
        Perform round-robin ranking of items based on their categories.

        Returns:
        list: Ranked list of item indices.
        """
        categories = defaultdict(deque)
        for item_id, category in self.item_dataframe.items():
            categories[category].append(item_id)
        
        if self.random_walk_score is not None:
            for category in categories:
                categories[category] = deque(
                    sorted(categories[category], 
                        key=lambda item: self.random_walk_score[item] if 0 <= item < len(self.random_walk_score) else float("-inf"), 
                        reverse=True)
                )

        result = []
        # Convert dict to deque of queues for easy round-robin access
        category_queues = deque(categories.values())
        # Perform round-robin selection
        while category_queues:
            # Get the next category's queue
            current_queue = category_queues.popleft()
            # Pop an item from this category and add to the result
            if current_queue:
                result.append(current_queue.popleft())
            # If there are still items in this category, add the queue back to the end
            if current_queue:
                category_queues.append(current_queue)
        return result

    def rank(self):
        """
        Rank items using the graph coloring method, with a fallback to round-robin ranking.

        Returns:
        list: List of ranked item IDs.
        """

        cache_key = self._generate_cache_key()

        # Check if the result is already in the cache
        if cache_key in ItemRanker.cache:
            return ItemRanker.cache[cache_key]

        order_target = self.solve_graph_coloring()
        # if cannot find a solution in graph coloring, using round_robin_rank.
        if len(order_target) == 0:
            ordered_item_ids = self.round_robin_rank()
        else:
            category_to_items = defaultdict(list)
            for item_id, category in self.item_dataframe.items():
                category_to_items[category].append(item_id)
            
            if self.random_walk_score is not None:
                for category in category_to_items:
                    category_to_items[category].sort(
                        key=lambda item: self.random_walk_score[item] if item < len(self.random_walk_score) else float("-inf"),
                        reverse=True
                    )

            ordered_item_ids = []
            # from color to item ids
            for category in order_target:
                if category in category_to_items and category_to_items[category]:
                    # Assign the next available item ID from this category
                    ordered_item_ids.append(category_to_items[category].pop(0))
                else:
                    # If no item ID is available for this category or the category does not match, append None
                    ordered_item_ids.append(None)
        # Store the result in cache
        ItemRanker.cache[cache_key] = ordered_item_ids

        return ordered_item_ids

    @classmethod
    def clear_cache(cls):
        """Clears the class-level cache."""
        cls.cache = {}
