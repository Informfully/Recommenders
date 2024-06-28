# Internal version. Please do not share publicly.

from __future__ import print_function
import numpy as np
from math import log

class GraphRec(object):

    def __init__(self, train_matrix):

        self.train_matrix = train_matrix

        block1 = np.concatenate((np.zeros((self.train_matrix.shape[0],
                                           self.train_matrix.shape[0])),
                                 self.train_matrix), axis = 1)

        block2 = np.concatenate((self.train_matrix.T,
                                 np.zeros((self.train_matrix.shape[1],
                                           self.train_matrix.shape[1]))), axis = 1)

        self.A = np.concatenate((block1, block2), axis=0)
        self.D = np.sum(self.A, 1)
        self.D[self.D == 0] = 0.0001

        self.P = 1.0 * self.A / self.D[:, None]
        self.P2 = None
        self.P3 = None

        self.num_u = train_matrix.shape[0]
        self.num_i = train_matrix.shape[1]

    def compute_p2_p3(self):
        self.P2 = np.dot(self.P, self.P)
        self.P3 = np.dot(self.P2, self.P)

    def _user_item_distance_broadening(self, user_pos, item_pos):
        diff = item_pos - user_pos
        return np.abs(diff / (self.max_pos - self.min_pos))

    def _user_item_distance_moderating(self, user_pos, item_pos):
        diff = item_pos - user_pos

        if (user_pos < 0 and diff <= 0):
            return 0.1

        elif ((user_pos > 0 and diff >= 0)):
            return 0.1

        else:
            abs_diff = np.abs(diff/(self.max_pos - self.min_pos))
            return abs_diff

    def _construct_item_user_distance_matrix(self, u_pos, i_pos, type):

        if (type == "moderating"):
            i_u_dist = np.abs(np.vectorize(self._user_item_distance_moderating)
                              (*np.meshgrid(i_pos, u_pos, indexing = 'xy')))

        elif (type == "broadening"):
            i_u_dist = np.abs(np.vectorize(self._user_item_distance_broadening)
                              (*np.meshgrid(i_pos, u_pos, indexing = 'xy')))

        upper_block = np.concatenate((np.zeros((self.num_u, self.num_u)), i_u_dist), axis = 1)

        lower_block = np.concatenate((np.zeros((self.num_i, self.num_u)),
                                      np.zeros((self.num_i, self.num_i))), axis = 1)

        item_user_dist_matrix = np.concatenate((upper_block, lower_block), axis = 0)

        return item_user_dist_matrix

    def train_RWE_ideology(self, u_pos, i_pos, b = 2, iters = 5, type = "moderating"):

        assert len(u_pos) == self.num_u
        assert len(i_pos) == self.num_i

        self.max_pos = max(np.max(i_pos), np.max(u_pos))
        self.min_pos = min(np.min(i_pos), np.min(u_pos))

        total_mass = np.zeros((self.P.shape[0], self.P.shape[1]))

        if (self.P2 is None or self.P3 is None):
            self.compute_p2_p3()

        item_user_distance_matrix = self._construct_item_user_distance_matrix(u_pos, i_pos, type)

        dist_reweigh = item_user_distance_matrix ** b

        P3 = self.P3.copy()

        for iter in range(0, iters):

            erase = P3 - (P3 * dist_reweigh)

            mass = P3 * dist_reweigh
            D_mod = np.sum(erase, 1)

            v_s = np.diag(D_mod)
            total_mass += P3
            total_mass -= erase

            P3 = np.dot(v_s, self.P3)

        recs = total_mass + erase

        return recs[:self.train_matrix.shape[0], self.train_matrix.shape[0]:]
