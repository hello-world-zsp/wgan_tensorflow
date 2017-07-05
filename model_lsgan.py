# -*-coding:utf-8-*-
import tensorflow as tf
import os
import time
from ops import *
from helper_functions import *

class LSGAN (object):
    def __init__(self, sess, img_size = 80, is_crop = True,
                 batch_size=100, sample_size = 64, output_size = 80,
                 y_dim=None, z_dim = 100, gf_dim=64, df_dim=16,
                 gfc_dim=1024, dfc_dim=1024, c_dim = 1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None,
                 num_filter = 16, leak = 0, stride = [1, 1, 1, 1],clip_min = -2, clip_max = 2):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.img_size = img_size
        self.sample_size = sample_size
        self.output_size = img_size

        self.y_dim = y_dim
        self.z_dim = z_dim          #initial noise for g_model

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.num_filter = num_filter
        self.leak = leak
        self.stride = stride
        self.alpha = 0
        self.alpha_step = 0.05
        self.clip_min = clip_min
        self.clip_max = clip_max


        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def generator(self, gdata):
        w = self.output_size[0]
        h = self.output_size[1]
        # s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
        w2, w4, w8, w16 = int(w / 2), int(w / 4), int(w / 8), int(w / 16)
        h2, h4, h8, h16 = int(h / 2), int(h / 4), int(h / 8), int(h / 16)
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(gdata, self.gf_dim * 4 * w8 * h8,
                                                   'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, w8, h8, self.gf_dim * 4])      # batch_size*5*5*512
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,[self.batch_size, w4, h4, self.gf_dim * 2],
                                                     name='g_h1',with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,[self.batch_size, w2, h2, self.gf_dim * 1],
                                                name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,[self.batch_size,w, h, self.c_dim],
                                                name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            # h4, self.h4_w, self.h4_b = deconv2d(h3,[self.batch_size, s, s, self.c_dim],
            #                                     name='g_h4', with_w=True)
            # temp1 = tf.nn.sigmoid(self.g_bn4(h3))
            temp1 = tf.nn.relu(self.g_bn4(h3))

            return temp1


    def discriminator(self,ddata, reuse=False):
        with tf.variable_scope("discriminator") as scope:  # 自己加的with，解决196行adam问题
            if reuse:
                # tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()  # sharing variables

            stddev = 0.002

            h0 = lrelu(conv2d(ddata, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            # return tf.nn.sigmoid(h4), h4

        return h4              # wgan,去掉sigmoid层

    def build_model(self):
        self.real_images = tf.placeholder(tf.float32, [self.batch_size]
                                              + [self.img_size[0], self.img_size[1], self.c_dim], name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')


        self.G = self.generator(self.z)
        self.D_real = self.discriminator(self.real_images)
        self.D_fake = self.discriminator(self.G, reuse=True)

        self.G_sum = image_summary("G", self.G)

        # lsgan loss
        self.g_loss = mse(self.D_fake, tf.ones_like(self.D_fake))
        self.d_loss = mse(self.D_real, tf.ones_like(self.D_real)) + mse(self.D_fake, tf.zeros_like(self.D_fake))

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        # wgan , 对D网络参数限幅
        self.d_vars_clip = [var.assign(tf.clip_by_value(var,self.clip_min,self.clip_max)) for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    # def train(sess, G, d_loss, d_vars, g_loss, g_vars, saver, c_dim=1):
    def train(self, config):
        # reals, noises = read_images2(self.c_dim,config)
        # reals = load_mnist('./MNIST_data')
        reals = load_imgs('../data/resize_simsun80_norm.npy')
        # reals = load_imgs('./data/0005/resize_rescale.npy')

        # wgan不用基于动量的optimizer
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate_g).minimize(self.g_loss,var_list=self.g_vars)
        # d_optim = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.GradientDescentOptimizer(config.learning_rate_g).minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.G_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        start_time = time.time()
        counter = 0

        sample_images = reals[0:self.batch_size]
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        model_name = "MYDCGAN.model"
        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join('./checkpoint', model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in range(config.epoch):

            num_batch = len(reals) // config.batch_size
            print('num_batch', num_batch)

            for idx in range(0, num_batch):
                # update D
                for _ in range(config.d_iters):
                    # np.random.shuffle(reals)
                    batch_images = reals[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                    self.sess.run(self.d_vars_clip)
                    _, errD, summary_str = self.sess.run([d_optim, self.d_loss, self.d_sum], feed_dict={self.real_images: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # update G
                for _ in range(config.g_iters):
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                    _, errG, summary_str = self.sess.run([g_optim,self.g_loss, self.g_sum],
                                                   feed_dict={self.real_images: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch,
                                                                                          idx, num_batch,
                                                                                          time.time() - start_time,
                                                                                          errD, errG))

                if counter == 1:
                    # save_images(sample_z, [10, 10], 'results/' + 'noise' + str(counter) + '.jpg')
                    save_images(sample_images, [6, 6], 'results/' + 'original' + str(counter) + '.jpg')
                if np.mod(counter, 200) == 1:
                    samples, loss1, loss2 = self.sess.run([self.G, self.d_loss,
                                                      self.g_loss], feed_dict={self.z: batch_z,
                                                                               self.real_images: sample_images})

                    save_images(samples, [6, 6], 'results/' + 'generate' + str(counter) + '.jpg')
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (loss1, loss2))

                if np.mod(counter, 500) == 2:
                    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=counter)


