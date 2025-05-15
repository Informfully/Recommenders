import numpy as np
import scipy.sparse as sp
from __future__ import print_function
from math import log
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack
from tqdm import tqdm


class GraphRec(object):
    """
     A graph-based recommender model that constructs a bipartite graph from a user-item interaction matrix.
     Attributes:
        -----------
        train_matrix : csr_matrix
            User-item interaction matrix.
        A : csr_matrix
            Adjacency matrix of the bipartite graph.
        P : csr_matrix
            Normalized adjacency matrix for one-hop probabilities.
        P_multi : dict
            Multi-hop probability matrices for multiple hop counts.
    """
    def __init__(self, train_matrix):
        """
        Constructs a bipartite graph of size (|U| + |V|) x (|U| + |V|) from a user-item interaction matrix, 
        where |U| represents the number of users and |V| represents the number of items. 

        Parameters:
        -----------
        train_matrix : array-like or csr_matrix
            User-item interaction matrix.
        """
        # Ensure the train_matrix is in CSR format and convert its data type to float32 for efficiency
        if not isinstance(train_matrix, csr_matrix):
            self.train_matrix = csr_matrix(train_matrix).astype(np.float32)
        else:
            self.train_matrix = train_matrix.astype(np.float32)

        # Create the adjacency matrix of the bipartite graph as CSR
        self.num_u = self.train_matrix.shape[0]
        self.num_i = self.train_matrix.shape[1]

        # Build adjacency matrix
        # Create block diagonal and off-diagonal blocks
        block1 = csr_matrix((self.num_u, self.num_u), dtype=np.float32)
        block2 = csr_matrix((self.num_i, self.num_i), dtype=np.float32)
        upper = hstack([block1, self.train_matrix])
        lower = hstack([self.train_matrix.T, block2])
        self.A = vstack([upper, lower])

        # Compute degree-normalized adjacency matrix
        self.D = np.array(self.A.sum(axis=1)).flatten()
        # Prevent division by zero
        self.D[self.D == 0] = 0.0001
        self.P = self.A.multiply(1.0 / self.D[:, None])
    
    def get_item_degrees(self):
        """
        Calculate the number of edges each item is connected to users.
        
        Returns:
        --------
        item_edges : array
            An array with the number of edges (interactions) for each item.
        """
        # Get the number of non-zero entries (edges) for each column (item) in the train_matrix
        # Count non-zero elements in each column
        item_edges = np.array(self.train_matrix.getnnz(axis=0)).flatten()
        return item_edges
    
    def performInitialHop(self):
        """
        Computes the 2-hop and 3-hop transition probability matrices.

        Parameters:
        -----------
        None.

        Output:
        -------
        Updates `self.P_multi` with 2-hop and 3-hop probability matrices.
        """
        # Perform matrix multiplication using CSR optimized methods
        P2 = self.P.dot(self.P).astype(np.float32)  # P squared
        self.P3 = P2.dot(self.P).astype(np.float32)  # P cubed
        return self.P3
        
    def predict_reranked_scores(self, user_idx, beta=0.7):
        """Calculate re-ranked scores using degree-based regularization
        
        Args:
            user_idx: Target user index
            beta: Weight parameter for degree regularization (0.7 as specified)
            
        Returns:
            ndarray: Re-ranked recommendation scores
        """
        # Get initial 3-hop probabilities 
        start_col = self.num_u
        recs = self.P3[user_idx, start_col:]
        recs_dense = recs.toarray().flatten() if isinstance(recs, sp.csr_matrix) else recs
        # print(recs_dense[recs_dense != 0])

        # Get item degrees and normalize
        item_degrees = self.get_item_degrees() 
        # print(f"item_degrees:{item_degrees}")

        # Avoid division by zero
        item_degrees[item_degrees == 0] = 0.0001
      
        # Popularity beta weighting
        popularity_penalty = 1 / (item_degrees ** beta)
        # print(f"popularity_penalty:{popularity_penalty}")

        # Perform element-wise multiplication with popularity_penalty
        reranked_scores = recs_dense * popularity_penalty
        
        return reranked_scores
