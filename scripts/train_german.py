from models.adv_model import AdvModel
from utils.gpu import set_gpu_config
import tensorflow as tf
import numpy as np
from data_parsing.german_data import get_german_data
import tensorflow.keras as keras

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()


num_epochs = 30
weights = [1, 1]

y_accuracy = []
s_accuracy = []
disparate_impacts = []
demographics = []
for model_id in range(20):
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

    my_model = AdvModel(input_size, [2, 2], weights)
    my_model.train((train_data, [train_labels_one_hot, train_protected_one_hot]), num_epochs=num_epochs)
    y_acc, di, dd, s_acc = my_model.evaluate(test_data, [test_labels_one_hot, test_protected_one_hot], [test_labels, test_protected])
    y_accuracy.append(y_acc)
    s_accuracy.append(s_acc)
    disparate_impacts.append(di)
    demographics.append(dd)

    print("Mean y_acc", np.mean(y_accuracy), np.std(y_accuracy))
    print("Mean s_acc", np.mean(s_accuracy), np.std(s_accuracy))
    print("Mean impact", np.mean(disparate_impacts), np.std(disparate_impacts))
    print("Mean disparity", np.mean(demographics), np.std(demographics))
    keras.backend.clear_session()
