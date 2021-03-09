"""
2020/10/14 writen by Runze Dong
fully-connection network
rewriten in 2020/10/16
a CNN-based network coming soon
"""

from fun_1_tf import *
from tensorflow.python.keras.layers import *
P_a = np.expand_dims(10**(np.linspace(-10, 10, 11)/10)/(beta_0*d_b**(-1*eta_b))*delta_[1, :], 0)
R_s1 = np.zeros([1, P_a.shape[1]])
path = './data/1/'
H_bta, H_eta, H_bka, H_eka = load_mat(path)
for i in range(0, P_a.shape[1]):
    P_a0 = P_a[0, i]
    H_bt1, H_et1 = H_bta[N:2*N, :, :], H_eta[N:2*N, :, :]
    H_bk1, H_ek1 = H_bka[N:2*N, :, :], H_eka[N:2*N, :, :]
    H_b = np.concatenate([np.real(H_bk1), np.imag(H_bk1)], 1)
    H_e = np.concatenate([np.real(H_ek1), np.imag(H_ek1)], 1)
    H_input = np.concatenate([H_b, H_e], 1)  # for input to the network
    P_a_input = np.expand_dims(np.repeat(P_a0, N), -1)
    # define the input shape of the network
    H_e_input = Input(name='H_e_input', shape=H_input.shape[1:3], dtype=tf.float32)  # shape withput batch_size
    H_bt_input = Input(name='H_bt_input', shape=H_bt1.shape[1:3], dtype=tf.complex64)
    H_et_input = Input(name='H_et_input', shape=H_et1.shape[1:3], dtype=tf.complex64)
    snrb = Input(name='snr1', shape=(1,), dtype=tf.complex64)
    snre = Input(name='snr2', shape=(1,), dtype=tf.complex64)
    P_a1 = Input(name='P_a1', shape=(1,), dtype=tf.float32)
    # t0 = Input(name='t0', shape=(1,), dtype=tf.complex64)
    delta = Input(name='delta', shape=(1,), dtype=tf.complex64)
    # define the network by functional model
    x = BatchNormalization()(H_e_input)
    x = Flatten()(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(256, activation='relu')(x)
    f_G = Dense(2*N_x*N_y*(t+1))(x)  # the front 2*N_x*N_y is for f
    # f, G = Lambda(f_G_and_power)([f_G, P_a1])
    f, G = Lambda(f_G_and_power)([f_G, P_a1])
    Loss = Lambda(Loss_calculating)([f, G, H_bt_input, H_et_input, snrb, snre, delta])
    model = Model(inputs=[H_e_input, H_bt_input, H_et_input, snrb, snre, delta, P_a1], outputs=Loss)

    ii = str(i+1)
    model.load_weights('./output/network_DSN/1/'+ii+'_trained.h5')
    model.summary()
    layer_model = Model(inputs=model.input, outputs=model.layers[7].output)  # f
    layer_model_0 = Model(inputs=model.input, outputs=model.layers[13].output)

    data_x = [H_input, H_bt1, H_et1, snr_b, snr_e, delta_, P_a_input]
    f, G = layer_model.predict(data_x)
    R_s_predict = layer_model_0.predict(data_x)
    io.savemat('./output/validate/1/f_'+ii+'.mat', {'f': f})
    io.savemat('./output/validate/1/G_'+ii+'.mat', {'G': G})

# --------------------
# validating R_sb and R_se
# --------------------
# R_sbs, R_ses, R_ss = np.zeros([N, 1]), np.zeros([N, 1]), np.zeros([N, 1])
# for ss in range(0, N):
#     H_bts = H_bt1[ss, :, :]
#     H_ets = H_et1[ss, :, :]
#     fs = np.expand_dims(f[ss, :], -1)
#     Gs = G[ss, :, :]
#     snrbs = np.expand_dims(snr_b[ss, :].astype(np.complex128), -1)
#     snres = np.expand_dims(snr_e[ss, :].astype(np.complex128), -1)
#     deltas = np.expand_dims(delta_[ss, :].astype(np.complex128), -1)
#     # R_sbs[ss, 0] = np.dot((snrbs*np.dot(H_bts, fs))
#     #         , np.conj(np.dot(H_bts, fs)).T)
#     R_sbs[ss, 0] = np.log(np.linalg.det(np.eye(N_b).astype(np.complex128)+np.dot(np.dot(snrbs*np.dot(H_bts, fs)
#             , np.conj(np.dot(H_bts, fs)).T), np.linalg.inv(np.dot(snrbs*np.dot(H_bts, Gs)
#             , np.conj(np.dot(H_bts, Gs)).T)+deltas*np.eye(N_b).astype(np.complex128)))).real)/np.log(2)
#     R_ses[ss, 0] = np.log(np.linalg.det(np.eye(N_e).astype(np.complex128)+np.dot(np.dot(snres*np.dot(H_ets, fs)
#             , np.conj(np.dot(H_ets, fs)).T), np.linalg.inv(np.dot(snres*np.dot(H_ets, Gs)
#             , np.conj(np.dot(H_ets, Gs)).T)+deltas*np.eye(N_e).astype(np.complex128)))).real)/np.log(2)
#     R_ss[ss, 0] = R_sbs[ss, 0]-R_ses[ss, 0]
#
# print(np.mean(-R_s_predict))
# print(np.mean(R_ss))



# sess = tf.InteractiveSession()
# print (x.eval())



