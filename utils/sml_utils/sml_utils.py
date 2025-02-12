import sklearn.metrics as skm
from river import forest


def create_arf():
    return forest.ARFClassifier(leaf_prediction="nb")

def test_cl(cl_table, models, X_test, y_test):
    for m in models:
        if "_ta" in m:
            models[m].reset_previous_data_points()
        for metric in ["accuracy", "kappa"]:
            cl_table[m][metric].append([])
        for task, (X_test_task, y_test_task) in enumerate(zip(X_test, y_test)):
            pred = []
            for x_cl, y_cl in zip(X_test_task, y_test_task):
                x_cl = {f"feat{i}": x_cl[i] for i in range(len(x_cl))}
                if type(models[m]) == list:
                    task = min(task, len(models[m]) - 1)
                    y_hat = models[m][task].predict_one(x_cl)
                else:
                    y_hat = models[m].predict_one(x_cl)
                y_hat = 0 if y_hat is None else y_hat
                pred.append(y_hat)
                if "_ta" in m:
                    models[m].update_inference(y_cl)
            cl_table[m]["accuracy"][-1].append(skm.accuracy_score(y_test_task, pred))
            cl_table[m]["kappa"][-1].append(skm.cohen_kappa_score(y_test_task, pred))
    return cl_table
