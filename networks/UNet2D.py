import numpy as np
import tensorflow as tf

from tensorflow import keras

tf.enable_eager_execution()


class conv_block(keras.layers.Layer):
    def __init__(self, ch_out):
        super(conv_block, self).__init__()
        self.conv = keras.Sequential([
            keras.layers.Conv2D(ch_out, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(ch_out, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.conv(x)
        return x


class up_conv(keras.layers.Layer):
    def __init__(self, ch_out):
        super(up_conv, self).__init__()
        self.up = keras.Sequential([
            keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            keras.layers.Conv2D(ch_out, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.up(x)
        return x


class Recurrent_block(keras.layers.Layer):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = keras.Sequential([
            keras.layers.Conv2D(ch_out, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(keras.layers.Layer):
    def __init__(self, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = keras.Sequential([
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        ])
        self.Conv_1x1 = keras.layers.Conv2D(ch_out, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Attention_block(keras.layers.Layer):
    def __init__(self, F_int):
        super(Attention_block, self).__init__()
        self.W_g = keras.Sequential([
            keras.layers.Conv2D(F_int, kernel_size=1, strides=1, padding='same'),
            keras.layers.BatchNormalization()
        ])
        self.W_x = keras.Sequential([
            keras.layers.Conv2D(F_int, kernel_size=1, strides=1, padding='same'),
            keras.layers.BatchNormalization()
        ])
        self.psi = keras.Sequential([
            keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('sigmoid')
        ])
        self.relu = keras.layers.ReLU()

    def call(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet2D(keras.models.Model):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNet2D, self).__init__()

        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.Conv1 = conv_block(ch_out=64)
        self.Conv2 = conv_block(ch_out=128)
        self.Conv3 = conv_block(ch_out=256)
        self.Conv4 = conv_block(ch_out=512)
        self.Conv5 = conv_block(ch_out=1024)

        self.Up5 = up_conv(ch_out=512)
        self.Up_conv5 = conv_block(ch_out=512)

        self.Up4 = up_conv(ch_out=256)
        self.Up_conv4 = conv_block(ch_out=256)

        self.Up3 = up_conv(ch_out=128)
        self.Up_conv3 = conv_block(ch_out=128)

        self.Up2 = up_conv(ch_out=64)
        self.Up_conv2 = conv_block(ch_out=64)

        self.Conv_1x1 = keras.layers.Conv2D(output_ch, kernel_size=1, strides=1)

    def call(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = keras.backend.concatenate((x4, d5), axis=-1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = keras.backend.concatenate((x3, d4), axis=-1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = keras.backend.concatenate((x2, d3), axis=-1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = keras.backend.concatenate((x1, d2), axis=-1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(keras.models.Model):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.Upsample = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.RRCNN1 = RRCNN_block(ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_out=1024, t=t)

        self.Up5 = up_conv(ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_out=512, t=t)

        self.Up4 = up_conv(ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_out=256, t=t)

        self.Up3 = up_conv(ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_out=128, t=t)

        self.Up2 = up_conv(ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_out=64, t=t)

        self.Conv_1x1 = keras.layers.Conv2D(output_ch, kernel_size=1, strides=1)

    def call(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = keras.backend.concatenate((x4, d5), axis=-1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = keras.backend.concatenate((x3, d4), axis=-1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = keras.backend.concatenate((x2, d3), axis=-1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = keras.backend.concatenate((x1, d2), axis=-1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(keras.models.Model):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.Conv1 = conv_block(ch_out=64)
        self.Conv2 = conv_block(ch_out=128)
        self.Conv3 = conv_block(ch_out=256)
        self.Conv4 = conv_block(ch_out=512)
        self.Conv5 = conv_block(ch_out=1024)

        self.Up5 = up_conv(ch_out=512)
        self.Att5 = Attention_block(F_int=256)
        self.Up_conv5 = conv_block(ch_out=512)

        self.Up4 = up_conv(ch_out=256)
        self.Att4 = Attention_block(F_int=128)
        self.Up_conv4 = conv_block(ch_out=256)

        self.Up3 = up_conv(ch_out=128)
        self.Att3 = Attention_block(F_int=64)
        self.Up_conv3 = conv_block(ch_out=128)

        self.Up2 = up_conv(ch_out=64)
        self.Att2 = Attention_block(F_int=32)
        self.Up_conv2 = conv_block(ch_out=64)

        self.Conv_1x1 = keras.layers.Conv2D(output_ch, kernel_size=1, strides=1)

    def call(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = keras.backend.concatenate((x4, d5), axis=-1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, x3)
        d4 = keras.backend.concatenate((x3, d4), axis=-1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, x2)
        d3 = keras.backend.concatenate((x2, d3), axis=-1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, x1)
        d2 = keras.backend.concatenate((x1, d2), axis=-1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(keras.models.Model):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.Upsample = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.RRCNN1 = RRCNN_block(ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_out=1024, t=t)

        self.Up5 = up_conv(ch_out=512)
        self.Att5 = Attention_block(F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_out=512, t=t)

        self.Up4 = up_conv(ch_out=256)
        self.Att4 = Attention_block(F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_out=256, t=t)

        self.Up3 = up_conv(ch_out=128)
        self.Att3 = Attention_block(F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_out=128, t=t)

        self.Up2 = up_conv(ch_out=64)
        self.Att2 = Attention_block(F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_out=64, t=t)

        self.Conv_1x1 = keras.layers.Conv2D(output_ch, kernel_size=1, strides=1)

    def call(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = keras.backend.concatenate((x4, d5), axis=-1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, x3)
        d4 = keras.backend.concatenate((x3, d4), axis=-1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, x2)
        d3 = keras.backend.concatenate((x2, d3), axis=-1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, x1)
        d2 = keras.backend.concatenate((x1, d2), axis=-1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


x = np.random.rand(12, 32, 32, 3).astype('float32')
z = np.zeros((12, 32, 32, 3)).astype('float32')

rnn_layer = RRCNN_block(3)

attn_block = Attention_block(3)

out = rnn_layer(x)

out = attn_block(x, x)

out = attn_block(x, z)







