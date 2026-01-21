import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def prepare_data_time_series(df, sequence_length=60, target_length=22):

    data_features = df.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_features)

    X = []
    Y = []
    all_dates = []

    for i in range(len(df) - sequence_length - target_length + 1):
        X.append(data_scaled[i: i + sequence_length])
        Y.append(data_scaled[i + sequence_length: i + sequence_length + target_length, 0])
        all_dates.append(df.index[i + sequence_length + target_length - 1])

    train_size = int(len(X) * 0.8)

    xt_train = X[:train_size]
    Yl_train = Y[:train_size]
    xt_val = X[train_size:]
    Yl_val = Y[train_size:]

    train_dates = all_dates[:train_size]
    val_dates = all_dates[train_size:]

    # Konversi list ke numpy array
    X_all, y_all = np.array(X), np.array(Y)
    X_train, y_train = np.array(xt_train), np.array(Yl_train)
    X_val, y_val = np.array(xt_val), np.array(Yl_val)

    return scaler, data_scaled, X_all, y_all, X_train, y_train, X_val, y_val, train_dates, val_dates


def inverse_scaler(data, scaler):
    num_features = scaler.scale_.shape[0]

    data_reshape = data.reshape(-1, 1)
    dummy_features = np.zeros((data_reshape.shape[0], num_features - 1))
    data_stack = np.hstack([data_reshape, dummy_features])
    data_inv = scaler.inverse_transform(data_stack)[:, 0]

    return data_inv


def predict_next_days(model, last_sequence_scaled, scaler):
    x = last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1])
    y_pred_scaled = model.predict(x)

    y_pred = inverse_scaler(y_pred_scaled, scaler)

    return y_pred

def predict_ensemble(train_models, scaler, last_sequence, meta_model, SEQUENCE_LENGTH=60):
    # Siapkan input dari sequence terakhir
    input_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

    model_predictions = []

    # Prediksi dari setiap base model
    for model in train_models:
        pred = model.predict(input_sequence, verbose=0)[0]
        model_predictions.append(pred)

    model_predictions_stacked = np.stack(model_predictions, axis=1)
    model_predictions_stacked = model_predictions_stacked.flatten().reshape(1, -1)

    # Prediksi gabungan lewat meta model (output 22 hari sekaligus)
    meta_pred = meta_model.predict(model_predictions_stacked, verbose=0)[0]


    meta_pred = meta_pred.reshape(-1, 1)

    meta_pred_inv = inverse_scaler(meta_pred, scaler)

    return meta_pred_inv, model_predictions

def create_df_predict_future(scaler, future_dates, model_names, y_true, future_predictions_individual, future_predictions_base):

    df_future = pd.DataFrame({
        'Tanggal': future_dates,
    })

    for i, preds in enumerate(future_predictions_individual):
        model_name = model_names[i]
        df_future[f'Prediksi {model_name}'] = inverse_scaler(preds, scaler)

    df_future['Prediksi Ensemble'] = future_predictions_base
    df_future['Harga Aktual'] = y_true

    return df_future

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def model_evaluate(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.4f}%")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

def create_df_predict_future(scaler, future_dates, model_names, y_true, future_predictions_individual, future_predictions_base):

    df_future = pd.DataFrame({
        'Tanggal': future_dates,
    })

    for i, preds in enumerate(future_predictions_individual):
        model_name = model_names[i]
        df_future[f'Prediksi {model_name}'] = inverse_scaler(preds, scaler)

    df_future['Prediksi Ensemble'] = future_predictions_base
    df_future['Harga Aktual'] = y_true

    return df_future
