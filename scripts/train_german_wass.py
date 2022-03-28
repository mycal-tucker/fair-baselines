import numpy as np

from data_parsing.german_data import get_german_data
from models import wass
import warnings
warnings.filterwarnings("ignore")

num_epochs = 10000

# for alpha in [1.0, 0.9, 0.5]:
#     for eta in [0.01, 0.001, 0.0001]:
for alpha in [0.5]:
    for eta in [.001]:
        for batch_size in [100]:
            y_accuracy = []
            s_accuracy = []
            disparate_impacts = []
            demographics = []
            for model_id in range(20):
                np.random.seed(model_id)
                train_data, train_labels, train_protected, test_data, test_labels, test_protected = get_german_data(
                    'data/german_credit_data.csv', wass_setup=True)
                input_size = train_data.shape[1]

                my_model = wass.Wass(input_size, alpha=alpha)

                my_model.train((train_data, train_labels, train_protected), num_epochs=num_epochs, test_data=(test_data, test_labels, test_protected))
                print("Done with iter", model_id)

                y_acc, di, dd, s_acc = my_model.evaluate((test_data, test_labels, test_protected))
                y_accuracy.append(y_acc)
                s_accuracy.append(s_acc)
                disparate_impacts.append(di)
                demographics.append(dd)
            print()
            print("For alpha", alpha)
            print("For eta", eta)
            print("Mean y_acc", np.mean(y_accuracy), np.std(y_accuracy))
            print("Mean s_acc", np.mean(s_accuracy), np.std(s_accuracy))
            print("Mean impact", np.mean(disparate_impacts), np.std(disparate_impacts))
            print("Mean disparity", np.mean(demographics), np.std(demographics))
            print()