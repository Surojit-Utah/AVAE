import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import sys
sys.path.append("..")
from loss import vae_loss
from util import utils
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class Train():

    def __init__(self, run_id, encoder, decoder, optimizer, lr_schedular, encoder_use_batch_norm, decoder_use_batch_norm, train_from_checkpoint,
                 epochs, update_t_stat_epoch_fraction, update_hyperpriors, shuffle_scatter_samples, dataloader, latent_dim, print_every_epoch, save_every_epoch, save_model_epochs,
                 kld_scalar, dataset_name):

        ############################
        # Autoencoder model
        # Optimizer and LR schedular
        # Checkpoint
        ############################
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_use_batch_norm = encoder_use_batch_norm
        self.decoder_use_batch_norm = decoder_use_batch_norm
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular
        self.train_from_checkpoint = train_from_checkpoint
        self.epochs = epochs
        self.cur_epoch = 0
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.load_run_id = None
        self.latent_dim = latent_dim
        self.kld_scalar = kld_scalar
        self.shuffle_scatter_samples = shuffle_scatter_samples

        #####################
        # Tensorflow datasets
        #####################
        self.dataloader = dataloader
        self.batch_size = self.dataloader.batch_size
        self.train_dataset = None  # Will be created later
        self.t_stat_dataset = None  # Will be created later
        self.val_dataset = self.dataloader.create_val_dataset()
        self.update_t_stat_epoch_fraction = update_t_stat_epoch_fraction

        ##################################################
        # Parametrs of the hyper-prior, Gamma distribution
        ##################################################
        self.update_hyperpriors = update_hyperpriors
        self.alpha0 = 0
        self.beta0 = np.zeros(self.latent_dim, dtype=np.float32)
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.beta_tensor = tf.compat.v1.get_variable('beta_arr', shape=(self.latent_dim), initializer=tf.constant_initializer(self.beta), trainable=False)
        self.beta_tensor_update = self.beta_tensor
        self.total_t_stat_samples = self.dataloader.t_stat_samples
        self.samples_t_stat = np.zeros((self.encoder.axis_samples, self.latent_dim), dtype=np.float32)
        self.mean_t_stat = np.zeros((self.encoder.axis_samples, self.latent_dim), dtype=np.float32)
        self.done_update_hyperpriors = False


        ##########
        # log data
        ##########
        self.dataset_name = dataset_name
        self.tb_log_dir = os.path.join("logs", self.dataset_name, "Dim_"+str(self.latent_dim), "Run_"+str(run_id), "tf_logs")
        self.summary_writer = tf.summary.create_file_writer(self.tb_log_dir)
        self.spec_model_dir, self.generated_image_dir, self.reconstructed_image_dir, \
        self.latent_repr_image_dir, self.save_model_dir = utils.create_log_directory(self.dataset_name, self.latent_dim, run_id)
        self.print_every_epoch = print_every_epoch
        self.save_every_epoch = save_every_epoch
        self.save_model_epochs = save_model_epochs
        # train statistics
        self.avg_ae_loss_train = 0
        self.avg_kld_loss_train = 0
        self.avg_reg_loss_train = 0
        self.avg_total_loss_train = 0
        # validation statistics
        self.avg_ae_loss_val = 0
        self.avg_kld_loss_val = 0
        self.avg_kld_loss_wo_const_val = 0
        self.avg_reg_loss_val = 0
        self.avg_total_loss_val = 0

        #################
        # Load checkpoint
        #################
        if self.train_from_checkpoint:
            self.load_checkpoint()


    def load_checkpoint(self):
        if self.train_from_checkpoint:
            self.load_run_id = input("Enter the run id: ")
            checkpoint_dir = os.path.join("logs", self.dataset_name, "Dim_"+str(self.latent_dim), "Run_"+str(self.load_run_id), "Models")
            status = self.model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            status.assert_existing_objects_matched()
            print("Loaded trained model parameters!!")


    def print_output(self):

        print("Epoch no                                 : ", self.cur_epoch)
        print("Training data reconstruction loss        : ", self.avg_ae_loss_train)
        print("Validation data reconstruction loss      : ", self.avg_ae_loss_val)
        print("Training data VAE KLD loss               : ", self.avg_kld_loss_train)
        print("Validation data VAE KLD loss             : ", self.avg_kld_loss_val)
        print("Validation data VAE KLD loss w/o const   : ", self.avg_kld_loss_wo_const_val)
        print("Training data Reg loss                   : ", self.avg_reg_loss_train)
        print("Validation data Reg loss                 : ", self.avg_reg_loss_val)
        print("Training data total loss                 : ", self.avg_total_loss_train)
        print("Validation data total loss               : ", self.avg_total_loss_val)
        print("Current learning rate                    : ", self.optimizer.learning_rate.numpy())

        # Writing in Tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Recons Loss', self.avg_ae_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val Recons Loss', self.avg_ae_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Train KLD Loss', self.avg_kld_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val KLD Loss', self.avg_kld_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Val KLD Loss w/o const', self.avg_kld_loss_wo_const_val, step=self.cur_epoch)
            tf.summary.scalar('Train Reg Loss', self.avg_reg_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val Reg Loss', self.avg_reg_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Train Total Loss', self.avg_total_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val Total Loss', self.avg_total_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Learning rate', self.optimizer.learning_rate.numpy(), step=self.cur_epoch)



    def plot_array_with_num(self, plot_array, save_img_path):

        plot_array = np.round(plot_array, 1)
        dimension = plot_array.shape[0]
        cols = min(dimension, 20)
        if (dimension % cols) > 0:
            rows = (dimension // cols) + 1
        else:
            rows = (dimension // cols)
        fig, ax = plt.subplots(rows, cols, squeeze=False)
        fig.set_size_inches(cols, rows)
        plt.axis("off")

        row_index = 0
        col_index = 0
        for i in range(dimension):
            ax[row_index, col_index].text(0.5, 0.5, str(plot_array[i, i]), ha="center", va="center", color="r")
            col_index += 1
            # adjust the row and col index
            if col_index == cols:
                col_index = 0
                row_index += 1

        fig.suptitle('Dimension-wise variance')
        fig.tight_layout()
        plt.savefig(save_img_path)
        plt.close(plt.gcf())


    def save_output(self):

        # Covariance of the mean representation
        samples = self.dataloader.x_val[:self.batch_size]
        val_encoder_output = self.encoder(samples, use_batch_norm=self.encoder_use_batch_norm, training=False)
        mean, log_var = tf.split(val_encoder_output, num_or_size_splits=2, axis=1)
        latent_vectors = mean.numpy()
        mean_covariance_matrix = np.cov(latent_vectors.T)
        diag_covariance_matrix = np.diagonal(mean_covariance_matrix)
        sorted_diag_covariance_matrix = np.sort(diag_covariance_matrix)[::-1]
        arg_sorted_diag_covariance_matrix = np.argsort(diag_covariance_matrix)[::-1]

        # Mean spread
        latent_image_path = os.path.join(self.latent_repr_image_dir, 'mean_spread_' + str(self.cur_epoch) + '.png')
        max_sigma = np.sqrt(sorted_diag_covariance_matrix[0])
        utils.diag_axis_splom(latent_vectors, latent_image_path, max_sigma)

        # Estimated Covariance matrix
        est_var = np.divide(self.beta_tensor_update.numpy(), self.alpha)
        plot_est_var = est_var[arg_sorted_diag_covariance_matrix]
        est_cov_mat = np.zeros((self.latent_dim, self.latent_dim))
        np.fill_diagonal(est_cov_mat, est_var)

        # Reconstructed Images
        samples = self.dataloader.x_val[:self.batch_size]
        val_encoder_output = self.encoder(samples, use_batch_norm=self.encoder_use_batch_norm, training=False)
        mean, log_var = tf.split(val_encoder_output, num_or_size_splits=2, axis=1)
        sample_z = self.encoder.sampler.get_sample([mean, log_var])
        val_decoder_output = self.decoder(sample_z, use_batch_norm=self.decoder_use_batch_norm, training=False)
        img_plot_input = samples

        # DSprites decoder produces logits
        if 'DSprites' in self.dataset_name:
            img_val_reconstructed = tf.math.sigmoid(val_decoder_output).numpy()
        else:
            img_val_reconstructed = val_decoder_output.numpy()

        if self.dataset_name=='CelebA':
            img_plot_input = (img_plot_input + 1) / 2
            img_val_reconstructed = (img_val_reconstructed + 1) / 2
        row_cnt = col_cnt = int(np.sqrt(self.batch_size))
        fig = utils.show_combined_images(img_plot_input, img_val_reconstructed, row_cnt, col_cnt * 2)
        reconstructed_image_path = os.path.join(self.reconstructed_image_dir, 'recons_' + str(self.cur_epoch) + '.png')
        plt.savefig(reconstructed_image_path)
        plt.close(plt.gcf())
        del samples

        # Generated Images
        samples = self.dataloader.x_val[:self.batch_size]
        val_encoder_output = self.encoder(samples, use_batch_norm=self.encoder_use_batch_norm, training=False)
        mean, log_var = tf.split(val_encoder_output, num_or_size_splits=2, axis=1)
        latent_vectors = mean.numpy()
        mean_vector = np.mean(latent_vectors, axis=0)
        sampled_z = np.zeros((self.batch_size, self.latent_dim))
        for dim_index in range(self.latent_dim):
            axis_std_dev = np.sqrt(est_cov_mat[dim_index, dim_index])
            sampled_z[:, dim_index] = mean_vector[dim_index] + \
                                      np.random.normal(loc=0, scale=1, size=self.batch_size)*axis_std_dev
        Gen_Test_Images = self.decoder(sampled_z, use_batch_norm=self.decoder_use_batch_norm, training=False)

        # DSprites decoder produces logits
        if 'DSprites' in self.dataset_name:
            Gen_Test_Images = tf.math.sigmoid(Gen_Test_Images).numpy()
        else:
            Gen_Test_Images = Gen_Test_Images.numpy()

        if self.dataset_name=='CelebA':
            Gen_Test_Images = (Gen_Test_Images + 1) / 2
        gen_test_image_path = os.path.join(self.generated_image_dir, "generated_" + str(self.cur_epoch) + ".png")
        fig = utils.show_images(Gen_Test_Images, row_cnt, col_cnt)
        plt.savefig(gen_test_image_path)
        plt.close(plt.gcf())
        del samples


    @tf.function
    def train_step_update(self, batch_x, beta_tensor, cur_beta_val):

        beta_tensor.assign(tf.cast(cur_beta_val, dtype=tf.float32))

        with tf.GradientTape(persistent=False) as tape:

            encoder_output = self.encoder(batch_x, use_batch_norm=self.encoder_use_batch_norm, training=True)
            mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
            # to be used in the reconstruction loss and the TC loss
            sample_z = self.encoder.sampler.get_sample([mean, log_var])

            # reconstruction loss
            decoder_output = self.decoder(sample_z, use_batch_norm=self.decoder_use_batch_norm, training=True)
            # autoencoder loss
            if 'DSprites' in self.dataset_name:
                ae_loss = vae_loss.autoencoder_ce_loss(batch_x, decoder_output)
            else:
                ae_loss = vae_loss.autoencoder_loss(batch_x, decoder_output)

            # sample based kld loss
            kld_loss = vae_loss.kld_loss(mean, log_var, self.alpha, beta_tensor)

            # total reg loss
            reg_loss = self.kld_scalar*kld_loss

            # total_loss
            total_loss = ae_loss + reg_loss

        model_grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(model_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return ae_loss, kld_loss, reg_loss, beta_tensor


    def train_model(self):

        self.lr_schedular.on_train_begin()
        for epoch in range(self.epochs):

            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            if epoch % self.update_t_stat_epoch_fraction == 0:
                self.train_dataset, self.t_stat_dataset = self.dataloader.create_t_stat_n_train_dataset()

            # epoch on training data
            avg_ae_loss_train, avg_kld_loss_train, avg_reg_loss_train, \
            avg_total_loss_train = self.run_an_epoch(self.train_dataset, epoch_no=epoch, mode='train')
            self.avg_ae_loss_train = avg_ae_loss_train
            self.avg_kld_loss_train = avg_kld_loss_train
            self.avg_reg_loss_train = avg_reg_loss_train
            self.avg_total_loss_train = avg_total_loss_train

            avg_ae_loss_val, avg_kld_loss_val, \
            avg_kld_loss_wo_const, avg_reg_loss_val, avg_total_loss_val = self.run_an_epoch(self.val_dataset, epoch_no=epoch, mode='val')
            self.avg_ae_loss_val = avg_ae_loss_val
            self.avg_kld_loss_val = avg_kld_loss_val
            self.avg_kld_loss_wo_const_val = avg_kld_loss_wo_const
            self.avg_reg_loss_val = avg_reg_loss_val
            self.avg_total_loss_val = avg_total_loss_val

            # Adjust the learning rate based on the validation loss
            self.lr_schedular.on_epoch_end(self.cur_epoch, avg_total_loss_val)

            ###################################
            # Logging and saving output, models
            ###################################
            if not epoch % self.print_every_epoch:
                self.print_output()

            if not epoch % self.save_every_epoch:
                self.save_output()

            if epoch in self.save_model_epochs:
                save_model_path = os.path.join(self.save_model_dir, "intermediate_model", "epoch_" + str(epoch), "ckpt")
                self.model_checkpoint.save(file_prefix=save_model_path)
                print("Model saved in file: %s" % save_model_path)

            print("Time to run an epoch: %.2fs" % (time.time() - start_time))

            self.cur_epoch += 1

        # save the model after training
        print("Training is successfully completed....")
        self.print_output()
        self.save_output()
        save_model_path = os.path.join(self.save_model_dir,  "ckpt")
        self.model_checkpoint.save(file_prefix = save_model_path)
        print("Model saved in file: %s" %save_model_path)


    def run_an_epoch(self, dataset, epoch_no, mode):

        avg_ae_loss = 0
        avg_kld_loss = 0
        avg_kld_loss_wo_const = 0
        avg_reg_loss = 0
        avg_total_loss = 0
        self.done_update_hyperpriors = False
        if mode=='train':
            # Iterate through the training dataset for training the model
            for step, x_batch in tqdm(enumerate(dataset)):

                if (self.done_update_hyperpriors == False) and (epoch_no%self.update_hyperpriors == 0):
                    print("\nUpdate Beta....")
                    self.samples_t_stat = np.zeros((self.encoder.axis_samples, self.latent_dim), dtype=np.float32)
                    self.mean_t_stat = np.zeros((self.encoder.axis_samples, self.latent_dim), dtype=np.float32)
                    num_batch_t_stat = self.encoder.axis_samples//self.batch_size

                    for t_stat_step, x_batch_t_stat in enumerate(self.t_stat_dataset):
                        if t_stat_step == num_batch_t_stat:
                            break
                        encoder_output = self.encoder(x_batch_t_stat, use_batch_norm=self.encoder_use_batch_norm,
                                                      training=False)
                        mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
                        start_index = t_stat_step * self.batch_size
                        end_index = (t_stat_step + 1) * self.batch_size
                        self.samples_t_stat[start_index:end_index] = self.encoder.sampler.beta_get_sample(
                            [mean, log_var]).numpy()
                        self.mean_t_stat[start_index:end_index] = mean.numpy()

                    cur_beta_update = np.sum(self.samples_t_stat**2, axis=0)/2
                    self.beta = self.beta0 + cur_beta_update
                    self.alpha = self.alpha0 + self.encoder.axis_samples/2
                    self.done_update_hyperpriors = True

                # Training the autoencoder along with VAE KLD loss
                ae_loss, kld_loss, reg_loss, beta_tensor_update = \
                    self.train_step_update(x_batch, self.beta_tensor, self.beta)

                self.beta_tensor_update = beta_tensor_update
                ae_loss, kld_loss, reg_loss = ae_loss.numpy(), kld_loss.numpy(), reg_loss.numpy()
                avg_ae_loss += ae_loss
                avg_kld_loss += kld_loss
                avg_reg_loss += reg_loss
                avg_total_loss += (ae_loss + reg_loss)

            avg_ae_loss = (avg_ae_loss / (step + 1))
            avg_kld_loss = (avg_kld_loss / (step + 1))
            avg_reg_loss = (avg_reg_loss / (step + 1))
            avg_total_loss = (avg_total_loss / (step + 1))

            return avg_ae_loss, avg_kld_loss, avg_reg_loss, avg_total_loss

        if mode=='val':
            for step, x_batch in enumerate(dataset):
                encoder_output = self.encoder(x_batch, use_batch_norm=self.encoder_use_batch_norm, training=False)
                mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
                sample_z = self.encoder.sampler.get_sample([mean, log_var])
                decoder_output = self.decoder(sample_z, use_batch_norm=self.decoder_use_batch_norm, training=False)

                if 'DSprites' in self.dataset_name:
                    ae_loss = vae_loss.autoencoder_ce_loss(x_batch, decoder_output)
                else:
                    ae_loss = vae_loss.autoencoder_loss(x_batch, decoder_output)
                ae_loss = ae_loss.numpy()

                # KL divergence loss w/o scalar
                cur_kld_loss = vae_loss.kld_loss(mean, log_var, self.alpha, self.beta_tensor_update)
                kld_loss = cur_kld_loss.numpy()

                cur_kld_loss_wo_const = vae_loss.kld_loss_wo_const(mean, log_var, self.alpha, self.beta_tensor_update).numpy()

                # Total reg loss
                reg_loss = self.kld_scalar * kld_loss

                avg_ae_loss += ae_loss
                avg_kld_loss += kld_loss
                avg_kld_loss_wo_const += cur_kld_loss_wo_const
                avg_reg_loss += reg_loss
                avg_total_loss += (ae_loss + reg_loss)

            avg_ae_loss = (avg_ae_loss / (step + 1))
            avg_kld_loss = (avg_kld_loss / (step + 1))
            avg_kld_loss_wo_const = (avg_kld_loss_wo_const/ (step + 1))
            avg_reg_loss = (avg_reg_loss / (step + 1))
            avg_total_loss = (avg_total_loss / (step + 1))

            print("Val Rec loss   : " + str(avg_ae_loss))
            print("Val KLD loss   : " + str(avg_kld_loss))
            print("Val Reg loss   : " + str(avg_reg_loss))
            print("Val Total loss : " + str(avg_total_loss))

            return avg_ae_loss, avg_kld_loss, avg_kld_loss_wo_const, avg_reg_loss, avg_total_loss