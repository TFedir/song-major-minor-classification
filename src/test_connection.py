import requests
from train_base import load_tracks

address = "http://localhost:9999/api/v1/base_model"
xgboost_address = "http://localhost:9999/api/v1/xgboost"
ab_address = "http://localhost:9999/api/v1/ab"


def test_ab():
    X, _,_ = load_tracks()
    first_row = X[0].tolist()
    payload = {"features": first_row, "user_id": 1}
    res = requests.post(ab_address, json=payload)
    assert res.ok

    prediction = res.json()
    print("Predykcja ab (uzytkownik 1):",prediction)

    payload = {"features": first_row, "user_id": 2}
    res = requests.post(ab_address, json=payload)
    assert res.ok

    prediction = res.json()
    print("Predykcja ab (uzytkownik 2):",prediction)


def test_request():
    X, _,_ = load_tracks()
    data = X[0:5].tolist()
    payload = {"features": data}
    res = requests.post(address, json=payload)
    assert res.ok

    prediction = res.json()
    print("predykcja modelu bazowego:",prediction)


def test_request_xgboost():
    X, _,_ = load_tracks()
    data = X[0:5].tolist()
    payload = {"features": data}
    res = requests.post(xgboost_address, json=payload)
    assert res.ok

    prediction = res.json()
    print("predykcja modelu bazowego:",prediction)