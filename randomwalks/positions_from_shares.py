import logging
import numpy as np

# Enable sessions (which are no longer part of TF2)
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# This is only a problem on the server under Linux. I excluded it for now...
# Import and set environment level to prevent AVX2 errors
#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# "True" skip adjustment below threshold and "False" goes through all score update iterations
IS_THRESHOLD_ACTIVATED = False

# Skip adjustment of article scores if updates are smaller than the threshold value
SCORING_THRESHOLD = 0.25

np.random.seed(0)
np.random.seed(0)

logger = logging.getLogger(__name__)

class IdeologyFromTweets(object):

    def __init__(self, 
                 matrix, 
                 user_positions,
                 rank = 1, lr = 0.05,
                 regularizer = 0.05,
                 do_normalize_matrix = False):

        logger.info("Logging Test")

        self.rank = rank
        self.lr = lr
        self.regularizer = regularizer
        self.steps = 1
        self.sess = tf.Session()

        assert matrix is not None
        
        if (do_normalize_matrix):
            self.ratings_matrix = self.normalize_matrices(matrix, "Ratings Matrix")
        
        else:
            self.ratings_matrix = matrix

        self.shape_ratings = self.ratings_matrix.shape

        assert (user_positions.shape[0] == self.ratings_matrix.shape[0])
        self.U = tf.convert_to_tensor(tf.reshape(user_positions, [self.shape_ratings[0], self.rank]))

    def normalize_matrices(self, matrix_ip, log_msg = ""):

        matrix = np.nan_to_num(matrix_ip)

        num_nz = np.count_nonzero(matrix)
        num_total_cells = matrix.shape[0] * matrix.shape[1]
        sum_nz = np.sum(matrix)

        logging.info("Normalization matrix %s: cells: %d, nz: %d, z: %d, sum: %d, density: %.2f, z/nz: %.2f"
                     % (log_msg, num_total_cells, num_nz, num_total_cells - num_nz, sum_nz,
                     1.0 * num_nz / num_total_cells, 1.0 * (num_total_cells - num_nz) / sum_nz))

        return (1.0 * (num_total_cells - num_nz) / sum_nz) * matrix

    def initialize_variables(self):

        self.V = tf.Variable(tf.random_uniform([self.rank, self.shape_ratings[1]], -1.0, 1.0))

        self.b_V = tf.Variable(tf.reshape(tf.zeros([self.shape_ratings[1]]), [1, self.shape_ratings[1]]))
        self.b_U = tf.Variable(tf.reshape(tf.zeros([self.shape_ratings[0]]), [self.shape_ratings[0], 1]))

        self.UV = tf.matmul(self.U, self.V) + self.b_V + self.b_U
        self.UV_exp = 1 / (1 + tf.exp(-self.UV))

        self.error = tf.reduce_sum(self.ratings_matrix * tf.log(self.UV_exp) + tf.log(1 - self.UV_exp))
        self.regularization = self.regularizer * tf.nn.l2_loss(self.U) + \
                              self.regularizer * tf.nn.l2_loss(self.V)

        self.loss = -(self.error + self.regularization)

        self.optimizer_U = tf.train.AdagradOptimizer(self.lr).minimize(
            self.loss, var_list=[self.b_U])
        self.optimizer_V = tf.train.AdagradOptimizer(self.lr).minimize(
            self.loss, var_list=[self.V, self.b_V])

        self.learnt_V = None
        self.learnt_U = None
        self.learnt_bU = None
        self.learnt_bV = None

        self.init = tf.global_variables_initializer()

    def train(self, steps, print_every_n_steps = 200):

        self.steps = steps

        with self.sess as sess:

            sess.run(self.init)

            lastRun = 100
            isActive = IS_THRESHOLD_ACTIVATED
            threshold = SCORING_THRESHOLD

            for i in range(self.steps):

                sess.run(self.optimizer_U)
                sess.run(self.optimizer_V)

                # Skip further training if the improvements over the iterations is lower than the threshold
                if (i % print_every_n_steps == 0):

                    #print("%d. Cost: %f" % (i, sess.run(self.loss)))
                    if (lastRun - sess.run(self.loss) < threshold) and (isActive == True):
                        break
                    
                    lastRun = sess.run(self.loss)

            self.learnt_V = sess.run(self.V)
            self.learnt_U = sess.run(self.U)
            self.learnt_bU = sess.run(self.b_U)
            self.learnt_bV = sess.run(self.b_V)

    def getU(self):
        return self.learnt_U

    def getV(self):
        return self.learnt_V

    def getb_u(self):
        return self.learnt_bU

    def getb_v(self):
        return self.learnt_bV
