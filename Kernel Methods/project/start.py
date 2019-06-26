# 3p
import numpy as np
import pandas as pd
# project
from kernels.kernel import Kernel, kernel_params
from classifiers.classifier import Classifier, classifier_params
from utils import is_result_saved, save_result, get_result


kgram = 9
params = [0.1, 0.01, 0.07]
kernel = Kernel("mismatch", {"kgram": kgram})

data = get_result("provided_data.pkl")
print("Computing needed matrices ...")
matrices = kernel.compute_nedeed_matrices(data)
print("Done!")


predictions = ((),)

for i in range(3):
    print("Training model for dataset {} ...".format(i))
    clf = Classifier("svm", {"_lambda":params[i]})
    clf.fit(matrices["Ktr{}".format(i)], matrices["Ytr{}".format(i)])
    predictions += (clf.predict(matrices["Kte{}".format(i)]),)

print("Saving predictions ...")
predictions = np.squeeze(np.hstack(predictions)).astype(int)
df = pd.DataFrame({'Bound': predictions, 'Id': np.arange(3000)})
df = df[['Id', 'Bound']]
filename = "Yte.csv"
df.to_csv(filename, index=False)
