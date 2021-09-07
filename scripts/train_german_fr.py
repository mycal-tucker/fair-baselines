from models.fr_train import FRTrain
from utils.gpu import set_gpu_config
import tensorflow as tf
import numpy as np
from data_parsing.german_data import get_german_data
import tensorflow.keras as keras

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()


num_epochs = 30
weights = [1, 0.2, 1]

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
    train_robustness = np.zeros_like(train_protected_one_hot)
    train_robustness[:, 1] = 1
    test_robustness = np.zeros_like(test_protected_one_hot)
    test_robustness[:, 1] = 1

    my_model = FRTrain([input_size, 2], [2, 2, 2], weights)
    # FIXME: need robust dataset for outputs
    my_model.train(([train_data, train_protected_one_hot], [train_labels_one_hot, train_protected_one_hot, train_robustness]), num_epochs=num_epochs)
    y_acc, di, dd, s_acc = my_model.evaluate([test_data, test_protected_one_hot], [test_labels_one_hot, test_protected_one_hot, test_robustness], [test_labels, test_protected, np.ones_like(test_protected)])
    y_accuracy.append(y_acc)
    s_accuracy.append(s_acc)
    disparate_impacts.append(di)
    demographics.append(dd)

    print("Mean y_acc", np.mean(y_accuracy), np.std(y_accuracy))
    print("Mean s_acc", np.mean(s_accuracy), np.std(s_accuracy))
    print("Mean impact", np.mean(disparate_impacts), np.std(disparate_impacts))
    print("Mean disparity", np.mean(demographics), np.std(demographics))
    keras.backend.clear_session()
