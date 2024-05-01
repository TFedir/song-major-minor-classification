from joblib import dump

def train(model,trainX, trainY, save_model=False, path=None):
    """
    model - model instance to train, needs to be sklearn interface (fit() method) 
    """
    model.fit(trainX, trainY)
    if save_model:
        dump(model, path)
    return model