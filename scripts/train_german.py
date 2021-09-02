from models.adv_model import AdvModel
from utils.gpu import set_gpu_config
import tensorflow as tf
import numpy as np
from data_parsing.german_data import get_german_data
import tensorflow.keras as keras

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

my_model = AdvModel(10, [2, 2])

num_epochs = 30
classification_weight = [10, .0]
proto_dist_weights = [1, .0]
feature_dist_weights = [1, .0]
disentangle_weights = [[0, 100], [0, 0]]
# disentangle_weights = [[0, 0], [0, 0]]
kl_losses = [0, 0]
batch_size = 32

y_accuracy = []
s_accuracy = []
disparate_impacts = []
demographics = []
for model_id in range(30):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    train_data, train_labels, train_protected, test_data, test_labels, test_protected = get_german_data('data/german_credit_data.csv', wass_setup=False)
    input_size = train_data.shape[1]

    # Create one-hot encodings of data
    train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes=2)
    train_protected_one_hot = keras.utils.to_categorical(train_protected)
    test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes=2)
    test_protected_one_hot = keras.utils.to_categorical(test_protected)

    num_protected_classes = train_protected_one_hot.shape[1]
    output_sizes = [2, num_protected_classes]
    train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
    test_outputs = [test_labels, test_protected]
    test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]
    mean_train_labels = np.mean(train_labels)
    print("Mean test rate", mean_train_labels)
    mean_test_rate = np.mean(test_labels)
    print("Mean test rate", mean_test_rate)