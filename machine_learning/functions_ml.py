from sklearn.metrics import confusion_matrix
def calculate_confusion_matrix_metrics(model, X_train, y_train):
    y_pred = model.predict(X_train)

    mat = confusion_matrix(y_train, y_pred)
    true_negatives = mat[0, 0]
    false_negatives = mat[1, 0]
    true_positives = mat[1, 1]
    false_positives = mat[0, 1]

    return {
        "confusion_matrix": mat,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "false_positives": false_positives
    }
def fpr(false_positives, true_negatives):
    return false_positives / (false_positives + true_negatives)



