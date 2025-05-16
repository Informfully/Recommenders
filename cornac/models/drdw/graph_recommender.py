from __future__ import print_function
import numpy as np
from math import log
from scipy.sparse import lil_matrix, csr_matrix
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
        self.D[self.D == 0] = 0.0001  # Prevent division by zero
        self.P = self.A.multiply(1.0 / self.D[:, None])
        self.P_multi = {}
        self.P_multi[1] = self.P

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
        P3 = P2.dot(self.P).astype(np.float32)  # P cubed
        self.P_multi[3] = P3

    def performMultiHop(self, m):
        """
        Computes the m-hop transition probability matrix using CSR matrix operations.

        Parameters:
        -----------
        m : int
            Number of hops to compute.

        Output:
        -------
        Returns the m-hop probability matrix.
        """
        current_hop = max(self.P_multi.keys())
        if m in self.P_multi:
            return self.P_multi[m]
        elif m > current_hop:
            
            remaining_hops = m - current_hop  # usually =2
            P_cur = self.P_multi[current_hop]
            for i in range(remaining_hops):

                P_cur = P_cur.dot(self.P).astype(np.float32)
                new_hop = current_hop + i + 1

                if new_hop % 2 == 1:
                    self.P_multi[new_hop] = P_cur
 
            return P_cur
        else:
            assert m <= current_hop



    def batched_multiply(self, A, B, batch_size=10000):
        """
        Performs batched sparse matrix multiplication without storing all intermediate results in memory.

        Parameters:
        -----------
        A : csr_matrix
            The left matrix.(sparse).
        B : csr_matrix
            The right matrix. (sparse).
        batch_size : int
            Number of rows to process per batch.

        Returns:
        --------
        csr_matrix
            The result of A * B computed in batches.
        """
        
        num_rows, num_cols = A.shape[0], B.shape[1]
        result_matrix = lil_matrix((num_rows, num_cols), dtype=np.float32)  # Initialize as LIL for efficient row-wise updates



        # print(f"Starting batch multiplication: {num_rows} rows, batch size = {batch_size}")

        # for start in range(0, num_rows, batch_size):
        for start in tqdm(range(0, num_rows, batch_size), desc="Processing Matrix Multiplication in Batches"):
            end = min(start + batch_size, num_rows)
            batch_result = A[start:end].dot(B)  # Multiply batch
            result_matrix[start:end] = batch_result
    

        return result_matrix.tocsr()
        