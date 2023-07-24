import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("results/result_Facenet.csv")
    tp_mean = round(df[df.decision == "Yes"].distance.mean(), 4)
    tp_std = round(df[df.decision == "Yes"].distance.std(), 4)
    fp_mean = round(df[df.decision == "No"].distance.mean(), 4)
    fp_std = round(df[df.decision == "No"].distance.std(), 4)
    print("positive: ", tp_mean, tp_std)
    print("negative: ", fp_mean, fp_std)
    sigma = 1
    threshold = round(tp_mean + sigma * tp_std, 4)
    print("statistical threshold, ", sigma, " sigma ", threshold)

    df[df.decision == "Yes"].distance.plot.kde()
    df[df.decision == "No"].distance.plot.kde()
    plt.legend(["Yes", "No"])
    plt.grid()
    plt.axhline(0,color='red')
    plt.axvline(0,color='red')
    plt.axvline(threshold, color="green")
    plt.show()
    sigma = 1
    threshold = round(tp_mean + sigma * tp_std, 4)
    print("statistical threshold ", threshold)

    df["prediction"] = "No" #init
    idx = df[df.distance <= threshold].index
    df.loc[idx, 'prediction'] = 'Yes'

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df.decision.values, df.prediction.values)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn)/(tn + fp +  fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("precision ", precision)
    print("recall ", recall)
    print("accuracy ", accuracy)