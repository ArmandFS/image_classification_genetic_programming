import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

datasets = ["f1", "f2"]

for dataSetName in datasets:
    print(f"\n=== Processing dataset: {dataSetName} ===")

    train_df = pd.read_csv(f"{dataSetName}_train_patterns.csv")
    test_df = pd.read_csv(f"{dataSetName}_test_patterns.csv")

    #Separate features and labels
    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    clf = LinearSVC(max_iter=2000)
    clf.fit(X_train, y_train)

    #Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    #Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")


