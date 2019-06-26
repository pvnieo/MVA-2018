# stdlib
import argparse
from time import time
# 3p
import numpy as np
import pandas as pd
# project
from kernels.kernel import Kernel, kernel_params
from classifiers.classifier import Classifier, classifier_params
from utils import is_result_saved, save_result, get_result


def additional_params(args):
    k_params, clf_params = {}, {}
    for param in kernel_params[args.kernel]:
        value = float(
            input(">>> Please, Give a value for kernel parameter `{}`: ".format(param)))
        k_params[param] = value
    for param in classifier_params[args.classifier]:
        value = float(
            input(">>> Please, Give a value for classifier parameter `{}`: ".format(param)))
        clf_params[param] = value
    return k_params, clf_params


def main():
    parser = argparse.ArgumentParser(
        description="Train and predict using a selected classifier and a kernel")
    parser.add_argument("-k", "--kernel", help="Choosen kernel",
                        type=str, default="gaussian", choices=["gaussian", "spectrum", "mismatch"])
    parser.add_argument("-c", "--classifier", help="Choosen classifier",
                        type=str, default="svm", choices=["svm", "lregression"])
    args = parser.parse_args()

    k_params, clf_params = additional_params(args)

    kernel = Kernel(args.kernel, k_params)

    kernel_name = "_".join(map(lambda x: str(x[0]) + "_" + str(x[1]), k_params.items()))
    clf_name = "_".join(map(lambda x: str(x[0]) + "_" + str(x[1]), clf_params.items()))
    filename = './dumps/' + "{}_{}_{}_{}.csv".format(args.kernel, kernel_name, args.classifier, clf_name)

    # Load Ks (kernel matrices) if computed, else compute them
    filename = "{}_{}.pkl".format(args.kernel, "_".join(
        map(lambda x: str(x[0]) + "_" + str(x[1]), k_params.items())))
    if is_result_saved(filename):
        print("Loading needed matrices ...")
        matrices = get_result(filename)
        print("Done!")
    else:
        since = time()
        data = get_result("provided_data.pkl")
        print("Computing needed matrices ...")
        matrices = kernel.compute_nedeed_matrices(data)
        print("Done!")
        print("Took {}s".format(int(time() - since)))
        save_result(filename, matrices)

    predictions = ((),)
    for i in range(3):
        print("Training model for dataset {} ...".format(i))
        clf = Classifier(args.classifier, clf_params)
        clf.fit(matrices["Ktr{}".format(i)], matrices["Ytr{}".format(i)])
        print("\tAccuracy: {} %".format(round(clf.evaluate(
            matrices["Kval{}".format(i)], matrices["Yval{}".format(i)]) * 100, 2)))
        predictions += (clf.predict(matrices["Kte{}".format(i)]),)

    print("Saving predictions ...")
    predictions = np.squeeze(np.hstack(predictions)).astype(int)
    df = pd.DataFrame({'Bound': predictions,
                       'Id': np.arange(3000)})
    df = df[['Id', 'Bound']]
    kernel_name = "_".join(map(lambda x: str(x[0]) + "_" + str(x[1]), k_params.items()))
    clf_name = "_".join(map(lambda x: str(x[0]) + "_" + str(x[1]), clf_params.items()))
    filename = './dumps/' + "{}_{}_{}_{}.csv".format(args.kernel, kernel_name, args.classifier, clf_name)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
