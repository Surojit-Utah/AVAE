import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import sys
sys.path.append("..")
from loss import avae_loss
from util import utils
import os
import glob
import shutil
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class Train():

    def __init__(self, run_id, encoder, decoder, optimizer, lr_schedular, encoder_use_batch_norm, decoder_use_batch_norm, train_data_noise, train_from_checkpoint,
                 epochs, update_KDE_epoch_fraction, update_q_iter_count, dataloader, latent_dim, bandwidth, max_cdf_epsilon, print_every_epoch, save_every_epoch, save_model_epochs, dataset_name, sigma_init_val):

        ########################################
        # Optimize data noise as scaling factor
        # OR
        # Use the proposed scaling factor
        ########################################
        if not train_data_noise:
            self.sigma_init_val = np.sqrt(sigma_init_val).astype(np.float32)
        else:
            self.sigma_init_val = 100.0
        self.sigma_sq = tf.compat.v1.get_variable('log_sigma', shape=(), initializer=tf.constant_initializer(self.sigma_init_val), trainable=train_data_noise)
        self.sigma_sq_update = self.sigma_sq

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
        if self.train_from_checkpoint:
            self.load_checkpoint()
        self.epochs = epochs
        self.cur_epoch = 0
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.load_run_id = None

        #####################
        # Tensorflow datasets
        #####################
        self.dataloader = dataloader
        self.batch_size = self.dataloader.batch_size
        self.train_dataset = None  # Will be created later
        self.kde_dataset = None  # Will be created later
        self.val_dataset = self.dataloader.create_val_dataset()
        self.update_KDE_epoch_fraction = update_KDE_epoch_fraction

        #######################################
        # KDE samples used for density estimate
        #######################################
        self.latent_dim = latent_dim
        self.q_samples_lagging_for_kde = np.zeros((self.dataloader.kde_samples, self.latent_dim), dtype=np.float32)
        self.update_q_iter_count = update_q_iter_count
        self.bandwidth = bandwidth
        self.Guassian_Prior_Std_Dev = np.sqrt(1 - self.bandwidth**2).astype(np.float32)
        self.max_cdf_epsilon = max_cdf_epsilon
        self.ENCODED_FLOAT_EPS = self.get_epsilon(self.bandwidth, self.latent_dim, max_cdf=self.max_cdf_epsilon)

        ##########
        # log data
        ##########
        self.dataset_name = dataset_name
        self.tb_log_dir = os.path.join("logs", self.dataset_name, "Run_"+str(run_id), "tf_logs")
        self.summary_writer = tf.summary.create_file_writer(self.tb_log_dir)
        self.spec_model_dir, self.generated_image_dir, self.reconstructed_image_dir, self.covariance_test_image_dir, \
        self.latent_repr_image_dir, self.save_model_dir = utils.create_log_directory(self.dataset_name, run_id)
        self.print_every_epoch = print_every_epoch
        self.save_every_epoch = save_every_epoch
        self.save_model_epochs = save_model_epochs
        # train statistics
        self.avg_ae_loss_train = 0
        self.avg_ae_loss_train_wo_const = 0
        self.avg_kld_loss_train = 0
        self.avg_total_loss_train = 0
        # validation statistics
        self.avg_ae_loss_val = 0
        self.avg_ae_loss_val_wo_const = 0
        self.avg_kld_loss_val = 0
        self.avg_total_loss_val = 0
        self.val_encoder_output = None

        self.best_val_loss = 1e09
        self.best_model_dir = os.path.join(self.save_model_dir, "best_model")
        self.best_model_path = os.path.join(self.best_model_dir, "ckpt")

    def get_epsilon(self, bandwidth, noise_dim, max_cdf=0.999):

        variance = np.round(bandwidth**2, 2)
        sq_dist_chi2 = stats.chi2.ppf(max_cdf, df=noise_dim, loc=0, scale=variance)
        epsilon = np.exp(-sq_dist_chi2/(2*variance)) * ((1/bandwidth)**noise_dim)

        return epsilon


    def load_checkpoint(self):
        if self.train_from_checkpoint:
            self.load_run_id = input("Enter the run id: ")
            checkpoint_dir = os.path.join("logs", self.dataset_name, "Run_"+str(self.load_run_id), "Models")
            status = self.model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            status.assert_existing_objects_matched()
            print("Loaded trained model parameters!!")
            # Overwrites the learning rate
            # optimizer.lr.assign(learning_rate)
            # print(optimizer.get_config()['learning_rate'])


    def print_output(self):
        # Norm of the KDE encodings
        latent_norm = np.linalg.norm(self.q_samples_lagging_for_kde, axis=1) ** 2

        # Probability w.r.t. the KDE
        prob_q = avae_loss.kde_for_samples(self.val_encoder_output, self.q_samples_lagging_for_kde, self.bandwidth)

        # Probability w.r.t. the standard normal
        prob_p = avae_loss.standard_normal_for_samples(self.val_encoder_output)

        print("Epoch no                                 : ", self.cur_epoch)
        print("Training data reconstruction loss        : ", self.avg_ae_loss_train)
        print("Validation data reconstruction loss      : ", self.avg_ae_loss_val)
        print("Training reconstruction loss W/O C       : ", self.avg_ae_loss_train_wo_const)
        print("Validation reconstruction loss W/O C     : ", self.avg_ae_loss_val_wo_const)
        print("Training data KLD loss                   : ", self.avg_kld_loss_train)
        print("Validation data KLD loss                 : ", self.avg_kld_loss_val)
        print("Training data total loss                 : ", self.avg_total_loss_train)
        print("Validation data total loss               : ", self.avg_total_loss_val)
        print("Reconstruction scaling factor            : ", self.sigma_sq_update.numpy())
        print("Current learning rate                    : ", self.optimizer.learning_rate.numpy())
        print("Prob w.r.t to the KDE                    : ")
        print("Mean prob w.r.t to the KDE               : ", np.mean(prob_q.numpy()))
        print("Min prob w.r.t to KDE                    : ", np.min(prob_q.numpy()))
        print("max prob w.r.t to KDE                    : ", np.max(prob_q.numpy()))
        print("Prob w.r.t to Std Normal                 : ")
        print("Mean prob w.r.t to Std Normal            : ", np.mean(prob_p.numpy()))
        print("Min prob w.r.t to Std Normal             : ", np.min(prob_p.numpy()))
        print("Max prob w.r.t to Std Normal             : ", np.max(prob_p.numpy()))
        print("Latent encoding statistics:")
        print("Min  distance                            : ", np.min(latent_norm))
        print("Max  distance                            : ", np.max(latent_norm))
        print("Mean distance                            : ", np.mean(latent_norm))

        # Writing in Tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Recons Loss', self.avg_ae_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val Recons Loss', self.avg_ae_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Train Recons Loss W/O Const', self.avg_ae_loss_train_wo_const, step=self.cur_epoch)
            tf.summary.scalar('Val Recons Loss W/O Const', self.avg_ae_loss_val_wo_const, step=self.cur_epoch)
            tf.summary.scalar('Train KLD Loss', self.avg_kld_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val KLD Loss', self.avg_kld_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Train Total Loss', self.avg_total_loss_train, step=self.cur_epoch)
            tf.summary.scalar('Val Total Loss', self.avg_total_loss_val, step=self.cur_epoch)
            tf.summary.scalar('Learning rate', self.optimizer.learning_rate.numpy(), step=self.cur_epoch)
            tf.summary.scalar('Sigma2', self.sigma_sq_update.numpy(), step=self.cur_epoch)


    def save_output(self):

        # Splom
        latent_image_path = os.path.join(self.latent_repr_image_dir, 'splom_' + str(self.cur_epoch) + '.png')
        utils.splom(self.q_samples_lagging_for_kde, latent_image_path)

        # Reconstructed Images
        val_encoder_output = self.encoder(self.dataloader.x_val[:self.batch_size],
                                          use_batch_norm=self.encoder_use_batch_norm, training=False)
        val_decoder_output = self.decoder(val_encoder_output, use_batch_norm=self.decoder_use_batch_norm,
                                          training=False)
        img_plot_input = self.dataloader.x_val[:self.batch_size]

        # DSprites decoder produces logits
        if self.dataset_name=='DSprites':
            img_val_reconstructed = tf.math.sigmoid(val_decoder_output).numpy()
        else:
            img_val_reconstructed = val_decoder_output.numpy()

        # Tanh in CelebA
        if self.dataset_name=='CelebA':
            img_plot_input = (img_plot_input + 1) / 2
            img_val_reconstructed = (img_val_reconstructed + 1) / 2

        row_cnt = col_cnt = int(np.sqrt(self.batch_size))
        fig = utils.show_combined_images(img_plot_input, img_val_reconstructed, row_cnt, col_cnt * 2)
        reconstructed_image_path = os.path.join(self.reconstructed_image_dir, 'recons_' + str(self.cur_epoch) + '.png')
        plt.savefig(reconstructed_image_path)
        plt.close(plt.gcf())

        # Covariance
        latent_vectors = val_encoder_output.numpy()
        Covariance_Matrix = np.cov(latent_vectors.T)
        fig = plt.figure()
        plt.imshow(Covariance_Matrix)
        plt.colorbar()
        covariance_test_image_path = os.path.join(self.covariance_test_image_dir,
                                                  "covraiance_image_" + str(self.cur_epoch) + ".png")
        plt.savefig(covariance_test_image_path)
        plt.close(plt.gcf())

        # Latent distance
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        n_bins = 2000
        data_dist_array = np.linalg.norm(self.q_samples_lagging_for_kde, axis=1) ** 2
        resultant_var = 1 - self.bandwidth ** 2
        ax1.hist(data_dist_array, bins=n_bins, histtype='stepfilled', color='b')
        ax1.set_ylabel('sample_count', color='b')

        max_cdf_distsq = stats.chi2.ppf(1 - 1e-06, df=self.latent_dim, loc=0, scale=resultant_var)
        input_plot_chi2 = np.linspace(0, max_cdf_distsq, 50000).astype(np.float64)
        plot_chi2_pdf = stats.chi2.pdf(input_plot_chi2, self.latent_dim, loc=0, scale=resultant_var).astype(np.float64)
        ax2.plot(input_plot_chi2, plot_chi2_pdf, 'r-', label='PDF_Chi2')
        ax2.set_xlabel('distance^2')
        ax2.set_ylabel('Chi2_PDF', color='r')

        covariance_test_image_path = os.path.join(self.covariance_test_image_dir,
                                                  "latent_encoding_" + str(self.cur_epoch) + ".png")
        plt.savefig(covariance_test_image_path)
        plt.close(plt.gcf())

        # Generated Images
        sampled_z = np.random.standard_normal(size=(self.batch_size, self.latent_dim)).astype(np.float32) * self.Guassian_Prior_Std_Dev
        Gen_Test_Images = self.decoder(sampled_z, use_batch_norm=self.decoder_use_batch_norm, training=False)

        # DSprites decoder produces logits
        if self.dataset_name=='DSprites':
            Gen_Test_Images = tf.math.sigmoid(Gen_Test_Images).numpy()
        else:
            Gen_Test_Images = Gen_Test_Images.numpy()

        # Tanh in CelebA
        if self.dataset_name=='CelebA':
            Gen_Test_Images = (Gen_Test_Images + 1) / 2

        gen_test_image_path = os.path.join(self.generated_image_dir, "generated_" + str(self.cur_epoch) + ".png")
        fig = utils.show_images(Gen_Test_Images, row_cnt, col_cnt)
        plt.savefig(gen_test_image_path)
        plt.close(plt.gcf())


    @tf.function
    def train_step_update(self, batch_x, q_samples_lagging_for_kde, q_samples_lagging_for_kde_bandwidth, sigma_sq, sigma_sq_val, ENCODED_FLOAT_EPS):

        sigma_sq.assign(tf.cast(sigma_sq_val, dtype=tf.float32))

        with tf.GradientTape(persistent=False) as tape:

            encoder_output = self.encoder(batch_x, use_batch_norm=self.encoder_use_batch_norm, training=self.encoder_use_batch_norm)
            decoder_output = self.decoder(encoder_output, use_batch_norm=self.decoder_use_batch_norm, training=self.decoder_use_batch_norm)

            # autoencoder loss
            if self.dataset_name=='DSprites':
                ae_loss, ae_loss_wo_const = avae_loss.autoencoder_ce_loss(batch_x, decoder_output, sigma_sq)
            else:
                ae_loss, ae_loss_wo_const = avae_loss.autoencoder_loss(batch_x, decoder_output, sigma_sq)

            # kld loss
            kld_loss = avae_loss.kld_loss_computation(encoder_output, q_samples_lagging_for_kde, q_samples_lagging_for_kde_bandwidth, ENCODED_FLOAT_EPS)

            # total_loss
            total_loss = ae_loss + kld_loss

        model_grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(model_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return ae_loss, ae_loss_wo_const, kld_loss, sigma_sq


    def train_model(self):

        self.lr_schedular.on_train_begin()
        for epoch in range(self.epochs):

            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            if epoch % self.update_KDE_epoch_fraction == 0:
                self.train_dataset, self.kde_dataset = self.dataloader.create_kde_n_train_dataset()

            # epoch on training data
            avg_ae_loss_train, avg_ae_loss_train_wo_const, avg_kld_loss_train, avg_total_loss_train = self.run_an_epoch(self.train_dataset, self.kde_dataset, mode='train')
            self.avg_ae_loss_train = avg_ae_loss_train
            self.avg_ae_loss_train_wo_const = avg_ae_loss_train_wo_const
            self.avg_kld_loss_train = avg_kld_loss_train
            self.avg_total_loss_train = avg_total_loss_train

            avg_ae_loss_val, avg_ae_loss_val_wo_const, avg_kld_loss_val, avg_total_loss_val, val_encoder_output = self.run_an_epoch(self.val_dataset, self.kde_dataset, mode='val')
            self.avg_ae_loss_val = avg_ae_loss_val
            self.avg_ae_loss_val_wo_const = avg_ae_loss_val_wo_const
            self.avg_kld_loss_val = avg_kld_loss_val
            self.avg_total_loss_val = avg_total_loss_val
            self.val_encoder_output = val_encoder_output

            # Adjust the learning rate based on the validation loss
            # Commented by SS on May 3, 2023
            # self.lr_schedular.on_epoch_end(self.cur_epoch, avg_total_loss_val)
            lr_sched_loss = self.avg_ae_loss_val_wo_const + self.avg_kld_loss_val
            self.lr_schedular.on_epoch_end(self.cur_epoch, lr_sched_loss)

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

            if self.avg_total_loss_val < self.best_val_loss:
                files = glob.glob(os.path.join(self.best_model_dir+"/*"))
                for f in files:
                    os.remove(f)
                self.model_checkpoint.save(file_prefix=self.best_model_path)
                print("Best model saved in file: %s" % self.best_model_path)
                self.best_val_loss = self.avg_total_loss_val


            print("Time to run an epoch: %.2fs" % (time.time() - start_time))

            self.cur_epoch += 1

        # save the model after training
        print("Training is successfully completed....")
        self.print_output()
        self.save_output()
        save_model_path = os.path.join(self.save_model_dir,  "ckpt")
        self.model_checkpoint.save(file_prefix = save_model_path)
        print("Model saved in file: %s" %save_model_path)


    def run_an_epoch(self, dataset, kde_dataset, mode):

        avg_ae_loss = 0
        avg_ae_loss_wo_const = 0
        avg_kld_loss = 0
        avg_total_loss = 0
        if mode=='train':
            # Iterate through the training dataset for training the model
            batches_per_epoch = self.dataloader.train_data_count//self.batch_size
            for step, x_batch in tqdm(enumerate(dataset)):

                # Minibatch index of the training data
                train_iter = int(self.cur_epoch * batches_per_epoch + step)

                # Update the KDE lagging samples
                if train_iter % self.update_q_iter_count == 0:
                    start_index = 0
                    # Iterate through the test dataset
                    for kde_step, x_batch_kde in enumerate(kde_dataset):
                        # Update the KDE samples in eval mode
                        kde_latent_encoding = self.encoder(x_batch_kde, use_batch_norm=self.encoder_use_batch_norm, training=False)
                        self.q_samples_lagging_for_kde[start_index:start_index + self.batch_size] = kde_latent_encoding.numpy()
                        start_index += self.batch_size

                # Training the autoencoder with kld loss
                ae_loss, ae_loss_wo_const, kld_loss, sigma_sq_update = self.train_step_update(x_batch,
                                                                                              self.q_samples_lagging_for_kde,
                                                                                              self.bandwidth,
                                                                                              self.sigma_sq,
                                                                                              self.sigma_init_val,
                                                                                              self.ENCODED_FLOAT_EPS)

                self.sigma_sq_update = sigma_sq_update
                ae_loss, ae_loss_wo_const, kld_loss = ae_loss.numpy(), ae_loss_wo_const.numpy(), kld_loss.numpy()
                avg_ae_loss += ae_loss
                avg_ae_loss_wo_const += ae_loss_wo_const
                avg_kld_loss += kld_loss
                avg_total_loss += (ae_loss + kld_loss)

            avg_ae_loss = (avg_ae_loss / (step + 1))
            avg_ae_loss_wo_const = (avg_ae_loss_wo_const / (step + 1))
            avg_kld_loss = (avg_kld_loss / (step + 1))
            avg_total_loss = (avg_total_loss / (step + 1))

            return avg_ae_loss, avg_ae_loss_wo_const, avg_kld_loss, avg_total_loss

        if mode=='val':
            for step, x_batch in enumerate(dataset):
                encoder_output = self.encoder(x_batch, use_batch_norm=self.encoder_use_batch_norm, training=False)
                decoder_output = self.decoder(encoder_output, use_batch_norm=self.decoder_use_batch_norm, training=False)
                if self.dataset_name == 'DSprites':
                    ae_loss, ae_loss_wo_const = avae_loss.autoencoder_ce_loss(x_batch, decoder_output, self.sigma_sq_update)
                else:
                    ae_loss, ae_loss_wo_const = avae_loss.autoencoder_loss(x_batch, decoder_output, self.sigma_sq_update)
                ae_loss, ae_loss_wo_const = ae_loss.numpy(), ae_loss_wo_const.numpy()

                # KL divergence loss
                kld_loss = avae_loss.kld_loss_computation(encoder_output,
                                                          self.q_samples_lagging_for_kde,
                                                          self.bandwidth,
                                                          self.ENCODED_FLOAT_EPS).numpy()

                avg_ae_loss += ae_loss
                avg_ae_loss_wo_const += ae_loss_wo_const
                avg_kld_loss += kld_loss
                avg_total_loss += (ae_loss + kld_loss)

            avg_ae_loss = (avg_ae_loss / (step + 1))
            avg_ae_loss_wo_const = (avg_ae_loss_wo_const / (step + 1))
            avg_kld_loss = (avg_kld_loss / (step + 1))
            avg_total_loss = (avg_total_loss / (step + 1))
            # The scaling factor is adjusted using the validation loss
            self.sigma_init_val = np.sqrt(avg_ae_loss_wo_const).astype(np.float32)

            return avg_ae_loss, avg_ae_loss_wo_const, avg_kld_loss, avg_total_loss, encoder_output
