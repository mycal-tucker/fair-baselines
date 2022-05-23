import numpy as np
from tensorflow import keras
import tensorflow as tf
from models.adv_model import AdvModel
from data_parsing.adult_data import get_adult_data
from utils.gpu import set_gpu_config


set_gpu_config()
tf.compat.v1.disable_eager_execution()

num_epochs = 30
weights = [1, 1]
output_sizes = [2, 2]


y_accuracy = []
s_accuracy = []
disparate_impacts = []
demographics = []
for model_id in range(20):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    train_data, train_labels_one_hot, train_protected_one_hot, test_data, test_labels_one_hot, test_protected_one_hot = get_adult_data('data/adult.csv', 'data/adult_test.csv')
    input_size = train_data.shape[1]
    protected_shape = train_protected_one_hot.shape

    # Create one-hot encodings of data
    train_protected = np.argmax(train_protected_one_hot, axis=1)
    test_labels = np.argmax(test_labels_one_hot, axis=1)
    test_protected = np.argmax(test_protected_one_hot, axis=1)

    my_model = AdvModel(input_size, output_sizes, weights)
    my_model.train((train_data, [train_labels_one_hot, train_protected_one_hot]), num_epochs=num_epochs)
    y_acc, di, dd, s_acc = my_model.evaluate(test_data, [test_labels_one_hot, test_protected_one_hot],
                                             [test_labels, test_protected])
    y_accuracy.append(y_acc)
    s_accuracy.append(s_acc)
    disparate_impacts.append(di)
    demographics.append(dd)

    print("Mean y_acc", np.mean(y_accuracy), np.std(y_accuracy))
    print("Mean s_acc", np.mean(s_accuracy), np.std(s_accuracy))
    print("Mean impact", np.mean(disparate_impacts), np.std(disparate_impacts))
    print("Mean disparity", np.mean(demographics), np.std(demographics))
    keras.backend.clear_session()
