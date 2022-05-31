from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Using Random forest classifier
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train, y_train)

    return rfc_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def slice_performance_test(df, model, encoder, lb, feature, cat_features):
    """
    A function that computes model metrics on slices of the data.

    Inputs
    ------
    df : The initial data to take the slice from 
    model : Random Forest Classifier model
        Trained machine learning model.
    encoder : One hot encoder 
    lb: the label binarizer
    feature : The feature column to perform the data slice on 
    cat_features: Categorical features 
    Returns
    -------

    """
    with open("slice_output.txt", "w") as text_file:
        for rows in df[feature].unique():
            # Get the sliced data and process it for inference 
            df_sliced = df[df[feature] == rows]
            X, y, _, _ = process_data(
                df_sliced, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            # Perform inference and get model metrics 
            preds = inference(model=model, X=X)
            precision, recall, fbeta = compute_model_metrics(y=y, preds=preds)

            # Output to txt file             
            text_file.write(f"Precision: {precision}\n")
            text_file.write(f"Recall: {recall}\n")
            text_file.write(f"Recall: {fbeta}\n")
            text_file.write("-----------------------\n")
        text_file.close()