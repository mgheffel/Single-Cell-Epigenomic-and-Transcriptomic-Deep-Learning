import os
import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class VAE:
    def __init__(self, input_shape,batch_size=256,optimizer=keras.optimizers.Adam(learning_rate=.001),epochs=50):

        self.mu = None
        self.sigma = None

        self.input_shape=input_shape
        self.batch_size=batch_size
        self.n_epochs=epochs
        self.optimizer=optimizer
        self.encoder=self.generate_encoder()
        self.encoder.compile(optimizer=optimizer,loss=self.vae_loss)
        self.sampler=self.perform_sampling()
        self.decoder=self.generate_decoder()
        self.decoder.compile(optimizer=optimizer,loss=self.vae_loss)
        self.kappa=1

    def vae_loss(self, y_true, y_pred):
        squared_error = tf.math.square(tf.math.subtract(tf.cast(y_true, tf.float64), tf.cast(y_pred, tf.float64)))
        recon = squared_error
        kl = K.mean(0.5*K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=1))
        recon = tf.cast(recon, dtype=tf.float64)  # Convert to double if needed
        self.kappa = tf.cast(self.kappa, dtype=tf.float64)  # Convert to double if needed
        kl = tf.cast(kl, dtype=tf.float64)
        return tf.math.add(tf.math.multiply(200,recon), tf.math.multiply(self.kappa,kl))

    def sample_z(self, args):
        vae_mu, vae_sigma = args
        eps = K.random_normal(shape=(K.shape(vae_mu)[0], K.shape(vae_mu)[1]), mean=0., stddev=1., seed=42)
        return vae_mu + K.exp(vae_sigma/2)*eps

    def kl_loss(self):
        return K.mean(0.5*K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=1))

    def generate_encoder(self):
        mC_input=keras.Input(shape=(self.input_shape,),name='mC_input')
        dense1=layers.Dense(128,activation='relu', name='dense1')(mC_input)
        dense2=layers.Dense(64,activation='relu', name='dense2')(dense1)
        vae_mu=layers.Dense(32,activation='linear')(dense2)
        vae_sigma=layers.Dense(32,activation='linear')(dense2)

        return Model(mC_input, (vae_mu, vae_sigma), name='encoder')

    def perform_sampling(self):
        vae_mu = keras.Input(shape=(32,))
        vae_sigma = keras.Input(shape=(32,))
        out=layers.Lambda(self.sample_z)([vae_mu, vae_sigma])
        return Model([vae_mu, vae_sigma], out, name='sampler')

    def generate_decoder(self):
        input_latent = keras.Input(shape=(32,))
        out_pred=layers.Dense(self.input_shape,activation='linear',name='out_pred')(input_latent)
        return Model(input_latent, out_pred, name='decoder')

    def train_vae(self,train_dataset,test_dataset):
        train_acc_metric = keras.metrics.MeanSquaredError()
        test_acc_metric = keras.metrics.MeanSquaredError()
        
        #early stopping
        patience = 10
        best_loss = float('inf')
        no_improvement_count = 0
        
        epochs = self.n_epochs
        for epoch in range(epochs):
                print("\nStart of epoch %d" % (epoch,))
                start_time = time.time()

                # Iterate over the batches of the dataset.
                for step, (x_batch_train) in enumerate(train_dataset):
                        with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
                                self.mu, self.sigma =self.encoder(x_batch_train, training=True)
                                latent = self.sampler([self.mu, self.sigma])
                                logits=self.decoder(latent, training=True)

                                loss_value = self.vae_loss(x_batch_train, logits)
                                # print(loss_value)
                        encoder_grads = encoder_tape.gradient(loss_value, self.encoder.trainable_weights)
                        decoder_grads = decoder_tape.gradient(loss_value, self.decoder.trainable_weights)
                        self.optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_weights))
                        self.optimizer.apply_gradients(zip(decoder_grads, self.decoder.trainable_weights))

                        # Update training metric.
                        train_acc_metric.update_state(x_batch_train, logits)

                        # Log every 31 batches.
                        if step % 31 == 0:
                            loss_scalar = tf.cast(loss_value, dtype=tf.float64).numpy()

                            print("Training loss (for one batch) at step ",step, tf.reduce_mean(loss_scalar))


                # Display metrics at the end of each epoch.
                train_acc = train_acc_metric.result()
                print("Reconstruction Training acc over epoch: %.4f" % (float(train_acc),))
                print("KL Divergence: %.10f" % (float(self.kl_loss())))

                # Reset training metrics at the end of each epoch
                train_acc_metric.reset_states()

                # Run a validation loop at the end of each epoch.
                for x_batch_test in test_dataset:
                        self.mu, self.sigma = self.encoder(x_batch_test, training=False)
                        latent = self.sampler([self.mu, self.sigma])
                        test_logits = self.decoder(latent, training=False)
                        
                        test_loss_value = self.vae_loss(x_batch_test, test_logits)
                        # Update val metrics
                        test_acc_metric.update_state(x_batch_test, test_logits)
                        
                test_acc = test_acc_metric.result()
                test_acc_metric.reset_states()
                print("Validation acc: %.4f" % (float(test_acc),))
                print('Validation loss: '+str(test_loss_value))

                if self.kappa < 0.95:
                        self.kappa += 0.25
                
                #early stopping
                test_loss=test_acc
#                 test_loss=tf.reduce_mean(loss_scalar)
                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print("Early stopping: No improvement for %d epochs." % patience)
                    break

    def predict(self,X):
        self.mu, self.sigma = self.encoder.predict(X)
        latent = self.sampler([self.mu, self.sigma])
        return self.decoder.predict(latent)

    def model_summary(self):
        print("Encoder Summary:")
        self.encoder.summary()

        print("\nSampler Summary:")
        self.sampler.summary()

        print("\nDecoder Summary:")
        self.decoder.summary()

    def visualize_model(self):
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)

    def correlation_accuracy(self, real_data):
        reconstructed_data = self.predict(real_data)
        corrs = 0
        for i in range(len(reconstructed_data)):
                corrs += pearsonr(reconstructed_data[i], real_data[i])[0]

        print(corrs/len(reconstructed_data))

    def mse_error(self, real_data):
        reconstructed_data = self.predict(real_data)
        mse = 0
        for i in range(len(reconstructed_data)):
                mse += mean_squared_error(reconstructed_data[i], real_data[i])

        print(mse/len(reconstructed_data))


