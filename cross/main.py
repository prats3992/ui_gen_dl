import argparse
from src.trainer import Trainer


def str2bool(v):
    return v.lower() in ("true")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)

    parser.add_argument("--epoch", type=int, default=1000,
                        help="The number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="The size of batch per gpu")
    parser.add_argument("--print_freq", type=int, default=1,
                        help="The number of image_print_freqy")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="The number of ckpt_save_freq")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of categories in the dataset"),

    parser.add_argument("--g_opt", type=str, default="adam",
                        help="learning rate for generator")
    parser.add_argument("--d_opt", type=str, default="adam",
                        help="learning rate for discriminator")
    parser.add_argument("--g_lr", type=float, default=0.00005,
                        help="learning rate for generator")
    parser.add_argument("--d_lr", type=float, default=0.0001,
                        help="learning rate for discriminator")
    parser.add_argument("--clip_val", type=float, default=0.01,
                        help="clips weights to this val")
    parser.add_argument("--n_critic", type=float, default=5,
                        help="number of disc updates per gen update")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.9,
                        help="beta2 for Adam optimizer")
    parser.add_argument("--gpl", type=float, default=2.0,
                        help="The gradient penalty lambda")

    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of noise vector")
    parser.add_argument("--image_size", type=int,
                        default=64, help="The size of image")
    parser.add_argument("--sample_num", type=int, default=5,
                        help="The number of sample images")

    parser.add_argument("--g_conv_filters", type=int,
                        default=16, help="basic filter num for generator")
    parser.add_argument("--g_conv_kernel_size", type=int,
                        default=4, help="basic kernel size for generator")
    parser.add_argument("--d_conv_filters", type=int,
                        default=16, help="basic filter num for disciminator")
    parser.add_argument("--d_conv_kernel_size", type=int,
                        default=4, help="basic kernel size for disciminator")

    parser.add_argument("--restore_model", action="store_true",
                        default=False, help="the latest model weights")
    parser.add_argument("--g_pretrained_model", type=str,
                        default=None, help="path of the pretrained model")
    parser.add_argument("--d_pretrained_model", type=str,
                        default=None, help="path of the pretrained model")

    parser.add_argument("--data_path", type=str,
                        default="./data/train/Akin_SAGAN_500/semantic")

    parser.add_argument(
        "--checkpoint_dir", type=str, default="./models/checkpoints", help="Directory name to save the checkpoints"
    )
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="Directory name to save the generated images")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory name to save training logs")
    parser.add_argument(
        "--sample_dir", type=str, default="./samples", help="Directory name to save the samples on training"
    )

    parser.add_argument("--category_file", type=str,
                        default="./resources/category.csv")

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)

    trainer.train()


if __name__ == "__main__":
    main()
