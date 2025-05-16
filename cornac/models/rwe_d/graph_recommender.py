from __future__ import print_function
import numpy as np
from math import log
from scipy.sparse import csr_matrix, hstack, vstack, diags

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
    

    def get_item_degrees(self):
        """
        Calculate the number of edges each item is connected to users.
        
        Returns:
        --------
        item_edges : array
            An array with the number of edges (interactions) for each item.
        """
        # Get the number of non-zero entries (edges) for each column (item) in the train_matrix
        # item_edges = np.array(self.train_matrix.getnnz(axis=0)).flatten()  # Count non-zero elements in each column
        # return item_edges
        return np.array(self.train_matrix.getnnz(axis=0)).astype(np.float32)
    
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
        
    

    def train_RWE_ideology(self,  beta = 0.7, iters=5):
        """ Train the ideology-aware recommender with a personalized random walk. """
    

        # total_mass = np.zeros_like(self.P.toarray(), dtype=np.float32)

        if not hasattr(self, 'P3'):
            self.performInitialHop()

    
        # popularity beta weighting
            
        item_degrees = self.get_item_degrees() 

        item_degrees[item_degrees == 0] = 1e-4   # Avoid division by zero

        print(f"item_degrees shape:{item_degrees.shape}")
      
        # popularity beta weighting
        popularity_penalty = 1 / (item_degrees ** beta)
        dist_reweigh = 1 - popularity_penalty

        print(f"dist_reweigh shape:{dist_reweigh.shape}")

        # i_u_dist  = np.tile(dist_reweigh, (self.num_u, 1))
        i_u_dist = csr_matrix(np.tile(dist_reweigh, (self.num_u, 1)), dtype=np.float32)
    

        # upper_block = np.concatenate((np.zeros((self.num_u, self.num_u)), i_u_dist), axis = 1)

        # lower_block = np.concatenate((np.zeros((self.num_i, self.num_u)),
        #                               np.zeros((self.num_i, self.num_i))), axis = 1)

        # item_user_dist_matrix = np.concatenate((upper_block, lower_block), axis = 0)
        upper_block = hstack([csr_matrix((self.num_u, self.num_u)), i_u_dist])
        lower_block = csr_matrix((self.num_i, self.num_u + self.num_i), dtype=np.float32)
        item_user_dist_matrix = vstack([upper_block, lower_block])


        print(f"item_user_dist_matrix shape:{item_user_dist_matrix.shape}")
        # total_mass = csr_matrix(self.P.shape, dtype=np.float32)
        
        total_mass = csr_matrix(self.P.shape, dtype=np.float32)
        P3 = self.P3.copy().tocsc()  # Con

        
        
        # P3 = self.P3.copy()

        for _ in range(iters):
            # erase = P3  * item_user_dist_matrix
            erase = P3.multiply(item_user_dist_matrix)  
            # D_mod = np.sum(erase, 1)
            D_mod = erase.sum(axis=1).A.flatten()
            # v_s = np.diag(D_mod)
            v_s = diags(D_mod, dtype=np.float32) 
            # total_mass += P3
            total_mass += P3 - erase
            # total_mass -= erase

            # P3 = np.dot(v_s, self.P3)
            P3 = v_s @ self.P3


        recs = total_mass + erase
        return recs[:self.num_u, self.num_u:].toarray()
