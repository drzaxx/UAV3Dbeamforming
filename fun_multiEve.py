import tensorflow as tf
import numpy as np
import scipy.io as io
from tensorflow.python.keras import *

N = 100000
t = 2  # (,*) dimension of G
# parameters
N_x, N_y, N_b, N_e = 4, 4, 6, 6
c_a = np.array([[0], [0], [0]])
c_b = np.array([[-100], [150], [200]])
# c_e = np.array([[100], [150], [220]])
c_e = io.loadmat('./c_e/c_e1.mat')['c__e']
c_e2 = io.loadmat('./c_e/c_e_multiEve2.mat')['c__e1']
c_e3 = io.loadmat('./c_e/c_e_multiEve3.mat')['c__e2']
beta_0_dB = -70
beta_0 = 10**(beta_0_dB/10)
eta_b, eta_e = 3.2, 3.2
d_b, d_e = np.linalg.norm(c_a-c_b), np.linalg.norm(c_a-c_e)
d_e2, d_e3 = np.linalg.norm(c_a-c_e2), np.linalg.norm(c_a-c_e3)
snr_b = beta_0*d_b**(-1*eta_b)
snr_b = np.expand_dims(np.repeat(snr_b, N), -1)
snr_e = beta_0*d_e**(-1*eta_e)
snr_e = np.expand_dims(np.repeat(snr_e, N), -1)
snr_e2, snr_e3 = beta_0*d_e2**(-1*eta_e), beta_0*d_e3**(-1*eta_e)
snr_e2, snr_e3 = np.expand_dims(np.repeat(snr_e2, N), -1), np.expand_dims(np.repeat(snr_e3, N), -1)
delta_ = np.expand_dims(np.repeat(1e-12, N), -1)


def load_mat(path):
    H_bt = io.loadmat(path + 'H_bt.mat')['H_bt']
    H_et = io.loadmat(path + 'H_et.mat')['H_et']
    H_bk = io.loadmat(path + 'H_bk.mat')['H_bk']
    H_ek = io.loadmat(path + 'H_ek.mat')['H_ek']
    return H_bt, H_et, H_bk, H_ek


def load_mat1(path):
    H_bt = io.loadmat(path + 'H_bt.mat')['H_bt'][N:2 * N, :, :]
    H_et = io.loadmat(path + 'H_et.mat')['H_et'][N:2 * N, :, :]
    H_bk = io.loadmat(path + 'H_bk.mat')['H_bk'][N:2 * N, :, :]
    H_ek = io.loadmat(path + 'H_ek.mat')['H_ek'][N:2 * N, :, :]
    return H_bt, H_et, H_bk, H_ek


def f_G_and_power(temp):
    f_G_temp, P_a0 = temp
    P_a0 = P_a0[0, :]
    f_G_temp = tf.nn.l2_normalize(f_G_temp, axis=1, epsilon=1e-10, name='nn_l2_norm')
    # f_G_temp = backend.dot(f_G_temp, tf.sqrt(P_a0))
    f_G_temp = tf.sqrt(P_a0)*f_G_temp
    f_0_real, f_0_imag = f_G_temp[:, 0:N_x*N_y], f_G_temp[:, N_x*N_y:2*N_x*N_y]
    G_0_real, G_0_imag = f_G_temp[:, 2*N_x*N_y:2*N_x*N_y+N_x*N_y*t],\
                         f_G_temp[:, 2*N_x*N_y+N_x*N_y*t:2*N_x*N_y+2*N_x*N_y*t]
    f = tf.complex(f_0_real, f_0_imag)
    G = tf.complex(G_0_real, G_0_imag)
    G1 = tf.concat(tf.split(tf.expand_dims(G, 2), num_or_size_splits=int(t), axis=1), 2)
    return f, G1
    # return f_0_imag


def Loss_calculating(temp):
    f, G, H_bt, H_et, H_et2, H_et3, snrb, snre, snre2, snre3, delta = temp
    snrb = snrb[0, :]
    snre, snre2, snre3 = snre[0, :], snre2[0, :], snre3[0, :]
    delta = delta[0, :]
    aa = backend.batch_dot(H_bt, f)
    aa1 = backend.batch_dot(tf.expand_dims(aa, 2), tf.transpose(tf.expand_dims(aa, 2), perm=[0, 2, 1], conjugate=True))
    bb = backend.batch_dot(H_bt, G)
    bb1 = backend.batch_dot(bb, tf.transpose(bb, perm=[0, 2, 1], conjugate=True))
    K_nb = snrb*bb1 + delta*tf.cast(tf.eye(N_b), tf.complex64)
    tempb = snrb*backend.batch_dot(aa1, tf.matrix_inverse(K_nb))
    aae = backend.batch_dot(H_et, f)
    aae1 = backend.batch_dot(tf.expand_dims(aae, 2), tf.transpose(tf.expand_dims(aae, 2), perm=[0, 2, 1], conjugate=True))
    bbe = backend.batch_dot(H_et, G)
    bbe1 = backend.batch_dot(bbe, tf.transpose(bbe, perm=[0, 2, 1], conjugate=True))
    K_ne = snre*bbe1 + delta*tf.cast(tf.eye(N_e), tf.complex64)
    tempe = snre*backend.batch_dot(aae1, tf.matrix_inverse(K_ne))
    R_sb = tf.math.log(tf.cast(tf.matrix_determinant(tf.cast(tf.eye(N_b), tf.complex64)+tempb), tf.float32))/tf.math.log(2.)
    R_se = tf.math.log(tf.cast(tf.matrix_determinant(tf.cast(tf.eye(N_e), tf.complex64)+tempe), tf.float32))/tf.math.log(2.)

    aaeS2 = backend.batch_dot(H_et2, f)
    aae2 = backend.batch_dot(tf.expand_dims(aaeS2, 2), tf.transpose(tf.expand_dims(aaeS2, 2), perm=[0, 2, 1], conjugate=True))
    bbe2 = backend.batch_dot(H_et2, G)
    bbe2 = backend.batch_dot(bbe2, tf.transpose(bbe2, perm=[0, 2, 1], conjugate=True))
    K_ne2 = snre2 * bbe2 + delta * tf.cast(tf.eye(N_e), tf.complex64)
    tempe2 = snre2 * backend.batch_dot(aae2, tf.matrix_inverse(K_ne2))
    R_se2 = tf.math.log(
        tf.cast(tf.matrix_determinant(tf.cast(tf.eye(N_e), tf.complex64) + tempe2), tf.float32)) / tf.math.log(2.)
    aaeS3 = backend.batch_dot(H_et3, f)
    aae3 = backend.batch_dot(tf.expand_dims(aaeS3, 2), tf.transpose(tf.expand_dims(aaeS3, 2), perm=[0, 2, 1], conjugate=True))
    bbe3 = backend.batch_dot(H_et3, G)
    bbe3 = backend.batch_dot(bbe3, tf.transpose(bbe3, perm=[0, 2, 1], conjugate=True))
    K_ne3 = snre3 * bbe3 + delta * tf.cast(tf.eye(N_e), tf.complex64)
    tempe3 = snre3 * backend.batch_dot(aae3, tf.matrix_inverse(K_ne3))
    R_se3 = tf.math.log(
        tf.cast(tf.matrix_determinant(tf.cast(tf.eye(N_e), tf.complex64) + tempe3), tf.float32)) / tf.math.log(2.)
    SSS = tf.reduce_max([R_se, R_se2, R_se3], 0)
    Loss = tf.expand_dims(R_sb-SSS, -1)
    # ss = tf.raw_ops.MatrixDeterminant(input=tf.cast(tf.eye(N_b), tf.complex64)+tempb)
    return -Loss


def self_defined_mean(y_true, y_pred):
    dd = backend.mean(y_pred, axis=-1)
    return dd


def expand_cnn(temp):
    out = tf.expand_dims(temp, -1)
    return out

# with tf.Session() as sess:
# ...     print(sess.run(tf.reduce_max(A)))

