import GPUtil
import catboost as cb
import xgboost as xgb
import lightgbm as lgbm
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression

gpu = len(GPUtil.getAvailable()) > 0

def lgbm_model():
    model = lgbm.LGBMRegressor(
                   objective='mse',
                   metric='mae',
                   subsample=0.7,
                   learning_rate=0.03,
                   n_estimators=10000,
                   n_jobs=-1
                   )
    return model

def xgb_model():
    model = xgb.XGBRegressor(
                   eval_metric='mae',
                   subsample=0.7,
                   tree_method='gpu_hist' if gpu else 'hist', # use GPU if available
                   learning_rate=0.03,
                   n_estimators=10000,
                   objective='reg:squarederror',
                    )
    return model

def cb_model():
    model = cb.CatBoostRegressor(
                   learning_rate=0.03,
                   iterations=10000,
                   loss_function='RMSE',
                   eval_metric='MAE',
                   use_best_model=True,
                   task_type='GPU' if gpu else 'CPU', # use GPU if available
                   early_stopping_rounds=100
                   )
    return model

def gru_model(lr=1e-3):
    inputs = layers.Input(shape=(365,))
    x = tf.expand_dims(inputs, axis=-1)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics='mean_absolute_error')
    return model

def lstm_model(lr=1e-3):
    inputs = layers.Input(shape=(365,))
    x = tf.expand_dims(inputs, axis=-1)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics='mean_absolute_error')
    return model

def lr_model():
    model = LinearRegression()
    return model

def mlp_model(lr=1e-3, loss='mse'):
    inputs = layers.Input(shape=(365,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    if loss == 'mse':
        metrics = 'mean_absolute_error'
    elif loss == 'mae':
        metrics = 'mean_squared_error'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss,
                  metrics=metrics)
    return model