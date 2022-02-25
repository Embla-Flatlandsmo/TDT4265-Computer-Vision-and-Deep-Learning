import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def create_comparison_plots(trainer_1: Trainer, trainer_2: Trainer, name:str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer_1.train_history["loss"], label="Training loss before", npoints_to_average=10)
    utils.plot_loss(trainer_2.train_history["loss"], label="Training loss after", npoints_to_average=10)
    utils.plot_loss(trainer_1.validation_history["loss"], label="Validation loss before")
    utils.plot_loss(trainer_2.validation_history["loss"], label="Validation loss after")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer_1.validation_history["accuracy"], label="Validation Accuracy before")
    utils.plot_loss(trainer_2.validation_history["accuracy"], label="Validation Accuracy after")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")

    _ , final_train_accuracy = compute_loss_and_accuracy(dataloaders[0])
    _ , final_val_accuracy = compute_loss_and_accuracy(dataloaders[1])
    _ , final_test_accuracy = compute_loss_and_accuracy(dataloaders[2])

    print("Train Accuracy: " + str(final_train_accuracy))
    print("Test Accuracy: " + str(final_test_accuracy))
    print("Validation Accuracy: " + str(final_val_accuracy))

    
if __name__ == "__main__":
    main()