import matplotlib.pyplot as plt
import os
import tensorflow as tf
from .data_generator import DataGenerator
from .model import create_generator, create_discriminator
import matplotlib
matplotlib.use('agg')


class Trainer:
    def __init__(self, args):
        """
        Trainer class to manage the training process with Wasserstein GAN.

        Args:
            args: A namespace object containing arguments for the trainer.
        """
        self.data_path = args.data_path
        args.image_size = (args.image_size, args.image_size)
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.num_classes = args.num_classes
        self.epochs = args.epoch

        # WGAN-specific hyperparameters
        self.clip_value = 0.01  # Clip weights to this range
        self.n_critic = 5  # Number of discriminator updates per generator update

        # Recommended learning rates for WGAN
        self.learning_rate_g = args.g_lr  # Lower learning rate for generator
        self.learning_rate_d = args.d_lr  # Same learning rate for discriminator

        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.sample_dir = args.sample_dir
        self.save_freq = args.save_freq
        self.print_freq = args.print_freq

        self.gen_loss_history = []
        self.disc_loss_history = []

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # Initialize data generator
        self.data_generator = self.get_data_generator()

        # Create models
        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()

        # Optimizers
        # Use RMSprop as recommended for WGAN
        self.generator_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate_g
        )
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate_d
        )

        # Checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5
        )

    def get_data_generator(self):
        """
        Initialize the DataGenerator for loading the dataset.

        Returns:
            DataGenerator: An instance of the DataGenerator class.
        """
        return DataGenerator(
            data_path=self.data_path,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

    def get_generator(self):
        """
        Create and return the generator model.

        Returns:
            tf.keras.Model: The generator model.
        """
        return create_generator(
            input_shape=self.image_size,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
        )

    def get_discriminator(self):
        """
        Create and return the discriminator model.

        Returns:
            tf.keras.Model: The discriminator model.
        """
        return create_discriminator(
            input_shape=self.image_size,
            num_classes=self.num_classes,
        )

    def save_samples(self, epoch):
        """
        Generate and save sample images, ensuring one from each category.

        Args:
            epoch (int): The current training epoch.
        """
        random_latent_vectors = tf.random.normal(
            shape=(self.num_classes, self.latent_dim)
        )
        class_labels = tf.range(self.num_classes)

        dummy_images = tf.zeros(
            shape=(self.num_classes, *self.image_size, 3), dtype=tf.float32)

        # Generate images
        generated_images, _, _ = self.generator(
            [dummy_images, random_latent_vectors, class_labels], training=False
        )

        # Save images
        for i in range(self.num_classes):
            image_path = os.path.join(self.sample_dir, f"epoch_{
                                      epoch}_class_{i}.png")
            tf.keras.utils.save_img(image_path, generated_images[i])

        print(f"Saved sample images for epoch {epoch} in {self.sample_dir}")

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): The current training epoch.
        """
        checkpoint_path = self.checkpoint_manager.save()
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    def train_step(self, real_images, labels):
        """
        Perform a single training step for WGAN.

        Args:
            real_images (tf.Tensor): Batch of real images.
            labels (tf.Tensor): Corresponding labels for the images.

        Returns:
            dict: Dictionary containing loss values for the step.
        """
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        random_labels = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        dummy_images = tf.zeros(
            shape=(batch_size, *self.image_size, 3), dtype=tf.float32)

        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images, _, _ = self.generator(
                [dummy_images, random_latent_vectors, random_labels], training=True
            )

            # Discriminator predictions
            real_output, _ = self.discriminator(
                [real_images, labels], training=True)
            fake_output, _ = self.discriminator(
                [generated_images, random_labels], training=True)

            # WGAN Discriminator loss
            disc_loss = tf.reduce_mean(
                fake_output) - tf.reduce_mean(real_output)

        # Compute and apply discriminator gradients
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))

        # Clip discriminator weights
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(
                var, -self.clip_value, self.clip_value))

        # Train Generator
        with tf.GradientTape() as gen_tape:
            # Regenerate images to ensure computational graph is updated
            generated_images, _, _ = self.generator(
                [dummy_images, random_latent_vectors, random_labels], training=True
            )

            # Generator predictions
            fake_output, _ = self.discriminator(
                [generated_images, random_labels], training=True)

            # WGAN Generator loss (minimize negative of discriminator's output)
            gen_loss = -tf.reduce_mean(fake_output)

        # Compute and apply generator gradients
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        self.gen_loss_history.append(gen_loss.numpy())
        self.disc_loss_history.append(disc_loss.numpy())

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    def plot_losses(self):
        """
        Plot generator and discriminator losses over training.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.gen_loss_history, label="Generator Loss")
        plt.plot(self.disc_loss_history, label="Discriminator Loss")
        plt.title("Training Losses")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "loss_plot.png"))
        plt.show()
        print(f"Loss plot saved at {os.path.join(
            self.result_dir, 'loss_plot.png')}")

    def train(self):
        """
        Train the models for the specified number of epochs.
        """
        dataset = self.data_generator.dataset

        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")

            # Train discriminator more frequently in WGAN
            for step, (real_images, labels) in enumerate(dataset):
                losses = self.train_step(real_images, labels)

                if step % self.print_freq == 0:
                    print(
                        f"Step {step}: "
                        f"Generator loss: {losses['gen_loss']:.4f}, "
                        f"Discriminator loss: {losses['disc_loss']:.4f}"
                    )

            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch)
                self.save_samples(epoch)
        self.plot_losses()
