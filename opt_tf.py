"""
2020/10/14 writen by Runze Dong
main file to the beamforming optimization
fully-connection network
rewriten in 2020/10/16
a CNN-based network coming soon
"""

from fun_1_tf import *
from tensorflow.python.keras.layers import *
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
P_a = np.expand_dims(10**(np.linspace(-10, 10, 11)/10)/(beta_0*d_b**(-1*eta_b))*delta_[1, :], 0)
R_s1 = np.zeros([1, P_a.shape[1]])
path = './data/1/'
H_bta, H_eta, H_bka, H_eka = load_mat(path)
for i in range(0, P_a.shape[1]):
# for i in range(0, 1):
    P_a0 = P_a[0, i]
    H_bt, H_et, H_bk, H_ek = H_bta[0:N, :, :], H_eta[0:N, :, :], H_bka[0:N, :, :], H_eka[0:N, :, :]
    H_b = np.concatenate([np.real(H_bk), np.imag(H_bk)], 1)
    H_e = np.concatenate([np.real(H_ek), np.imag(H_ek)], 1)
    H_input = np.concatenate([H_b, H_e], 1)  # for input to the network
    # define the input shape of the network
    H_e_input = Input(name='H_e_input', shape=H_input.shape[1:3], dtype=tf.float32)  # shape without batch_size
    H_bt_input = Input(name='H_bt_input', shape=H_bt.shape[1:3], dtype=tf.complex64)
    H_et_input = Input(name='H_et_input', shape=H_et.shape[1:3], dtype=tf.complex64)
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
    model.compile(optimizer='adam', loss=self_defined_mean)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    y_train = np.zeros([N, 1])
    P_a_input = np.expand_dims(np.repeat(P_a0, N), -1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00005)
    ii = str(i+1)
    print(ii)
    checkpoint = callbacks.ModelCheckpoint('./output/network_DSN/1/'+ii+'_trained.h5', monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min')
    model.fit(x=[H_input, H_bt, H_et, snr_b, snr_e, delta_, P_a_input], y=y_train, batch_size=8192,
              epochs=100, verbose=1, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    model.load_weights('./output/network_DSN/1/'+ii+'_trained.h5')
    model.summary()
    H_bt1, H_et1, H_bk1, H_ek1 = H_bta[N:2*N, :, :], H_eta[N:2*N, :, :], H_bka[N:2*N, :, :], H_eka[N:2*N, :, :]
    H_b = np.concatenate([np.real(H_bk1), np.imag(H_bk1)], 1)
    H_e = np.concatenate([np.real(H_ek1), np.imag(H_ek1)], 1)
    H_input = np.concatenate([H_b, H_e], 1)  # for input to the network
    # R_s = -model.predict([H_input, tf.cast(H_bt1, tf.complex64), tf.cast(H_et1, tf.complex64),
    #                       tf. cast(snr_b, tf.complex64), tf.cast(snr_e, tf.complex64), tf.cast(delta_, tf.complex64)])
    R_s = -model.predict([H_input, H_bt1, H_et1, snr_b, snr_e, delta_, P_a_input])
    R_s1[0, i] = np.mean(R_s)

io.savemat('./output/R_s/R_s_py1.mat', {'R_s_py': R_s1})
print(R_s1)
