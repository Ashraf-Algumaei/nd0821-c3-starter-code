import pytest
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from pytest_mock import MockerFixture


@pytest.fixture
def data():
    return pd.read_csv('./data/census_clean.csv')


@pytest.fixture
def train(data):
    train, _ = train_test_split(data, test_size=0.20)
    return train


@pytest.fixture
def test(data):
    _, test = train_test_split(data, test_size=0.20)
    return test    


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def encoder_lb(train, cat_features):
    _, _, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return (encoder, lb) 


@pytest.fixture
def rfc_model():
    return RandomForestClassifier()


@pytest.fixture
def mock_train_model(mocker: MockerFixture):
    return mocker.patch('starter.ml.model.train_model').return_value


@pytest.fixture
def mock_inference_result(mocker: MockerFixture):
    return mocker.patch('starter.ml.model.inference', return_value = np.ones(6513)).return_value


def test_train_model(cat_features, train, rfc_model):
    # GIVEN
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # WHEN
    model = train_model(X_train, y_train)

    # THEN
    assert type(model) == type(rfc_model)


def test_inference(cat_features, test, encoder_lb, mock_train_model):
    # GIVEN
    encoder, lb = encoder_lb
    X_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # WHEN
    inference(mock_train_model, X_test)

    # THEN
    mock_train_model.predict.assert_called_with(X_test) 


def test_compute_model_metrics(cat_features, test, encoder_lb, mock_inference_result):
    # GIVEN
    encoder, lb = encoder_lb
    _, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # WHEN
    precision, recall, fbeta = compute_model_metrics(y_test, mock_inference_result)
    
    # THEN
    assert precision is not None
    assert recall is not None
    assert fbeta is not None

