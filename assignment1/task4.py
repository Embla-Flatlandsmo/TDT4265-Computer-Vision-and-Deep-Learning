import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task4a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # (task 3c)
    predictions = model.forward(X)

    chosen_class = np.zeros_like(predictions)
    chosen_class[np.arange(len(predictions)), predictions.argmax(1)]=1
    correct = np.count_nonzero(targets==chosen_class)
    accuracy = correct / (targets.shape[0]*targets.shape[1])
    return accuracy

class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        #  (task 3b)
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w -= self.learning_rate*self.model.grad
        loss = cross_entropy_loss(Y_batch, outputs)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    # Plotting of softmax weights (Task 4b)
    # Shape of weights: [785, num_outputs]
    weight = model.w
    fig = plt.figure(figsize=(10,2))

    for i in range(0,10):
        sub = fig.add_subplot(2,10, i+1)
        sub.set_title(str(i))
        sub.axis('off')
        class_weights = weight[0:784,i]
        current_picture = np.reshape(class_weights, [28,28])
        sub.imshow(current_picture, interpolation='nearest', cmap="gray")

    weight = model1.w

    for i in range(0,10):
        sub = fig.add_subplot(2,10, 10+i+1)
        sub.set_title(str(i))
        sub.axis('off')
        class_weights = weight[0:784,i]
        current_picture = np.reshape(class_weights, [28,28])
        sub.imshow(current_picture, interpolation='nearest', cmap="gray")
    fig.tight_layout()
    plt.savefig("task4b_softmax_weight.png")
    plt.show()
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    plt.ylim([0.8, .99])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")

    l2_lambdas = [2, .2, .02, .002]
    l2_norms = np.empty(0)

    for l2_lambda in l2_lambdas:
        model2 = SoftmaxModel(l2_reg_lambda=l2_lambda)
        trainer2 = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
        )
        train_history_reg02, val_history_reg02 = trainer2.train(num_epochs)
        l2_norms = np.append(l2_norms, (np.linalg.norm(model2.w))) #apparently the default for linalg.norm is the l2 norm.
        utils.plot_loss(val_history_reg02["accuracy"], "Validation Accuracy, $\lambda$ = "+str(l2_lambda))
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.ylabel("$L_2$ norm length")
    plt.xlabel("Regularization strength $\lambda$")
    plt.plot(l2_lambdas, l2_norms, 'ro')
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()

