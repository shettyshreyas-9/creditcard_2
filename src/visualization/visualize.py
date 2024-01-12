import pathlib
import joblib
import sys
import yaml

import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model,X,y,split,live,save_path):

    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """

    predictions_by_class = model.predict_proba(X)
    predictions= predictions_by_class[:,1]

     # Use dvclive to log a few simple metrics...
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)


    # live.summary creates just the metrics.json file with summary

    if not live.summary:
        live.summary = {"avg_prec":{},"roc_auc":{}}
    live.summary["avg_prec"][split] = avg_prec
    live.summary["roc_auc"][split] = roc_auc 

     # ... and plots...
    # ... like an roc plot...
    live.log_sklearn_plot("roc",y, predictions,name= f"roc/{split}")

    # ... and precision recall plot...
    # ... which passes `drop_intermediate=True` to the sklearn method...
    live.log_sklearn_plot(
        "precision_recall",
        y,
        predictions,
        name=f"prc/{split}",
        drop_intermediate=True,
    )
    # ... and confusion matrix plot
    live.log_sklearn_plot(
        "confusion_matrix",
        y,
        predictions_by_class.argmax(-1),
        name=f"cm/{split}",
    )

    
    
def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir= curr_dir.parent.parent.parent

    # load the model
    model_path= sys.argv[1]
    model = joblib.load(model_path)

    # load the model
    input_file= sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix()+ '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)


    TARGET='Class'

    test_data = pd.read_csv(data_path + '/test.csv')
    X_test = test_data.drop(TARGET,axis=1)
    y_test= test_data[TARGET]

    # Evaluate test dataset
    with Live(output_path, dvcyaml=False) as live:
        evaluate(model, X_test, y_test, "test",live,output_path)


if __name__ == "__main__":
    main()