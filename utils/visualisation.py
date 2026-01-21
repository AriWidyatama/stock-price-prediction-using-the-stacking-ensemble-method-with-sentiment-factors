import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def inverse_scaler(data, scaler):
    num_features = scaler.scale_.shape[0]

    data_reshape = data.reshape(-1, 1)
    dummy_features = np.zeros((data_reshape.shape[0], num_features - 1))
    data_stack = np.hstack([data_reshape, dummy_features])
    data_inv = scaler.inverse_transform(data_stack)[:, 0]

    return data_inv

def plot_data(df, TICKER):
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Close'], label='Harga Penutupan (Close)', color='blue')
    plt.title(f"Pergerakan Harga Saham {TICKER}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (IDR)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotGraphsVal(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()

def plotGraphs(history, metric):
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric])
    plt.show()

def plot_predictions(data, scaler, train_dates, val_dates,
                     pred_train, pred_val,
                     future_dates, future_predictions, ticker):

    plt.figure(figsize=(14, 8))

    # Plot data historis (aktual)
    plt.plot(data.index, data['Close'], label='Harga Aktual', color='blue', linewidth=2)

    train_last = pred_train[:, -1].reshape(-1, 1)
    val_last = pred_val[:, -1].reshape(-1, 1)

    # Plot prediksi train dan val
    plt.plot(train_dates, inverse_scaler(train_last, scaler), label='Prediksi (Train)', color='green', linewidth=2)
    plt.plot(val_dates, inverse_scaler(val_last, scaler), label='Prediksi (Val)', color='red', linewidth=2, linestyle='--')

    # Plot prediksi masa depan
    plt.plot(future_dates, future_predictions, label='Prediksi Masa Depan', color='orange', linewidth=2)

    # Garis vertikal hari terakhir aktual
    latest_date = data.index[-1]
    plt.axvline(x=latest_date, linestyle='--', label='Hari Ini')

    # Judul dan label
    plt.title(f'Prediksi Harga Saham {ticker} - 22 Hari ke Depan', fontsize=16)
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ensemble_predictions(data, scaler, train_dates, val_dates,
                              individual_predictions_train, individual_predictions_val,
                              ensemble_predictions_train, ensemble_predictions_val,
                              future_dates, future_predictions, ticker):

    plt.figure(figsize=(14, 8))

    # Plot data historis (aktual)
    plt.plot(data.index, data['Close'], label='Harga Aktual', color='blue', linewidth=2)

    # Plot prediksi dari model-model individual
    colors = ['lightgray', 'darkgray', 'silver', 'gray']
    # Untuk model individual
    for i, (model_name, preds_train) in enumerate(individual_predictions_train.items()):
        # Plot prediksi train
        plt.plot(train_dates, inverse_scaler(preds_train[:, -1], scaler),
                label=f'{model_name} (Train)', color=colors[i % len(colors)],
                alpha=0.4, linewidth=1, linestyle='--')

        # Plot prediksi validasi jika ada
        if model_name in individual_predictions_val:
            last_step_preds = individual_predictions_val[model_name][:, -1]
            last_step_preds_inverse = inverse_scaler(last_step_preds, scaler)
            plt.plot(val_dates, last_step_preds_inverse,
                    label=f'{model_name} (Val)', color=colors[i % len(colors)],
                    alpha=0.7, linewidth=1, linestyle='-.')

    # Plot ensemble train dan val
    plt.plot(train_dates, inverse_scaler(ensemble_predictions_train[:, -1], scaler), label='Ensemble (Train)', color='green', linewidth=2)
    plt.plot(val_dates, inverse_scaler(ensemble_predictions_val[:, -1], scaler), label='Ensemble (Val)', color='red', linewidth=2, linestyle='--')

    # Plot prediksi masa depan
    plt.plot(future_dates, future_predictions, label='Prediksi masa depan', color='orange', linewidth=2)

    # Garis vertikal untuk hari terakhir aktual
    latest_date = data.index[-1]
    plt.axvline(x=latest_date, color='black', linestyle='--', label='Hari Ini')

    # Judul dan label
    plt.title(f'Prediksi Ensemble Harga Saham {ticker}',
              fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Harga Saham', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_future_predictions(future_dates, future_predictions_base):
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_predictions_base, marker='o', color='orange', label='Prediksi Harga Masa Depan')
    plt.title('Prediksi Harga Saham 30 Hari ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Saham (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_models_future_comparison(df, scaler, future_dates, model_names, y_true, future_predictions_individual, future_predictions_base, ticker):

    data = df.tail(60)

    plt.figure(figsize=(14, 7))

    # Plot data historis
    plt.plot(data.index, data['Close'], label='Historis Aktual', color='blue', linewidth=2)

    # Warna untuk masing-masing base model
    colors = ['red', 'purple', 'brown']

    # Loop berdasarkan index
    for i, preds in enumerate(future_predictions_individual):
        model_name = model_names[i]
        plt.plot(
            future_dates,
            inverse_scaler(preds, scaler),
            label=f'Prediksi {model_name}',
            color=colors[i % len(colors)],
            linewidth=2
        )

    plt.plot(future_dates, y_true, label='Harga Aktual', color='green', linewidth=2)

    plt.plot(future_dates, future_predictions_base, label='Prediksi Ensemble', color='orange', linewidth=2)

    # Garis pemisah antara data historis dan prediksi
    latest_date = data.index[-1]
    plt.axvline(x=latest_date, color='black', linestyle='--', linewidth=1.5, label='Hari Ini')

    # Judul dan styling
    plt.title(f'Perbandingan Hasil Prediksi Harga Saham {ticker} Setiap Model', fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Harga Saham', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_only_future_comparison(scaler, future_dates, model_names, y_true, future_predictions_individual, future_predictions_base, ticker):

    plt.figure(figsize=(14, 7))

    # Warna untuk masing-masing base model
    colors = ['red', 'purple', 'brown']

    # Loop berdasarkan index
    for i, preds in enumerate(future_predictions_individual):
        model_name = model_names[i]
        plt.plot(
            future_dates,
            inverse_scaler(preds, scaler),
            label=f'Prediksi {model_name}',
            color=colors[i % len(colors)],
            linewidth=2
        )

    plt.plot(future_dates, y_true, label='Harga Aktual', color='green', linewidth=2)

    plt.plot(future_dates, future_predictions_base, label='Prediksi Ensemble', color='orange', linewidth=2)

    # Judul dan styling
    plt.title(f'Perbandingan Hasil Prediksi Harga Saham {ticker} Setiap Model', fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Harga Saham', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()