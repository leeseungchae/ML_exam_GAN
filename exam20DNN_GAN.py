import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './DNN_out'
img_shape = (28,28,1)
epochs =100000
batch_size = 128
noise = 100
sample_interval = 100

(x_train, _) ,(_,_)  = mnist.load_data()
print(x_train.shape)

x_train = x_train / 127.5 -1  # -1 ~ 1   #희미한 값은 버리는되 약간 살리는 느낌?
x_train = np.expand_dims(x_train , axis=3)  # 차원늘리기  3차원
print(x_train.shape)

generator =  Sequential()
generator.add(Dense(128 , input_dim=noise))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

lrelu = LeakyReLU(alpha=0.01)
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                      optimizer='adam',metrics=['accuracy'])

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1 ))   # 0으로 채움 1차원 (128,1)

fake = np.zeros((batch_size, ))   #1으로 채움 1차원  (128,0)
discriminator.trainable=False
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0] , batch_size) #0~59999 int값 출력 (batch_size 만큼)  #중복허용(부트스트랩) 모아놓은거 베깅
    real_imgs = x_train[idx]

    z = np.random.normal(0, 1 ,(batch_size, noise) )   #평균이 0이고 표준편차 1
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)  #묶음 단위로 학습
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss , d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    discriminator.trainable=False  #학습취소

    if epoch <2000 :
        if epoch % 4 ==0:
            z = np.random.normal(0, 1, (batch_size, noise))
            #gan 모델 학습
            gan_hist = gan_model.train_on_batch(z, real)  #잡음 , real
    else:
        if epoch % 2 == 0:
            z = np.random.normal(0, 1, (batch_size, noise))
            # gan 모델 학습
            gan_hist = gan_model.train_on_batch(z, real)  # 잡음 , real



    if epoch % sample_interval  == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0,1,(row * col , noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row,col,figsize=(row,col), sharey=True, sharex=True)

        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cont,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR , 'img--{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()





















