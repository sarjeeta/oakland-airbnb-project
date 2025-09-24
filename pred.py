import math
import xgboost as xgb


def input_room(X, roomtype):
    list = [col for col in X.columns if col.startswith('room_type')]
    X[list] = 0
    X[roomtype] = 1
    return X

def input_area(X, areatype):
    list = [col for col in X.columns if col.startswith('neighbourhood')]
    X[list] = 0
    X[areatype] = 1
    return X

def input_amen(X, amenities):
    X[amenities] = 1
    return X

def predict_xgb(X, filename):
    model = xgb.XGBRegressor()
    model.load_model(filename)
    cols_when_model_builds = model.get_booster().feature_names
    X = X[cols_when_model_builds]
    y_pred = model.predict(X)
    y_pred = math.exp(y_pred)
    return round(y_pred, 3)
    
    

