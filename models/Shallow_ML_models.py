def execute_shallow_model(model=None, x_train=None, y_train=None, x_test=None):
    model.fit(x_train, y_train)

    best_params = model.best_params_
    best_estimator = model.best_estimator_

    y_pred = best_estimator.predict(x_test)  # test data
    x_pred = best_estimator.predict(x_train)  # training data

    return x_pred, y_pred, best_estimator, best_params
