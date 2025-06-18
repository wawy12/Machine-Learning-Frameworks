import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Permute, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

def create_cnn_model(input_shape, initial_learning_rate, total_steps):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,  
        decay_steps=total_steps,  
        alpha=0.0  
    )
    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),  
        Conv1D(36, kernel_size=3, padding='same', kernel_regularizer=regularizers.l1(3e-5)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        Conv1D(56, kernel_size=3, padding='same', kernel_regularizer=regularizers.l1(1e-4)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        MaxPooling1D(pool_size=2),
        Conv1D(36, kernel_size=3, padding='same', kernel_regularizer=regularizers.l1(6e-3)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(2e-3)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(4e-3)),
        Dense(1) 
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model

def create_cnn_lstm_model(input_shape, initial_learning_rate, total_steps):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,  
        alpha=0.0  
    )
    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),  
        Conv1D(56, kernel_size=3, padding='same', kernel_regularizer=regularizers.l1(7e-5)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        MaxPooling1D(pool_size=2),
        Permute((2, 1)),  
        LSTM(40, return_sequences=True, kernel_regularizer=regularizers.l1(2e-3)),
        Dropout(0.4),
        LSTM(8, return_sequences=True, kernel_regularizer=regularizers.l1(7e-4)), 
        Dropout(0.3),
        LSTM(72, return_sequences=False, kernel_regularizer=regularizers.l1(9e-5)), 
        Dropout(0.3),
        Flatten(),
        Dense(96, activation='relu', kernel_regularizer=regularizers.l2(3e-5)),
        Dense(1) 
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model

def create_lstm_model(input_shape, initial_learning_rate, total_steps):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,  
        decay_steps=total_steps,  
        alpha=0.0
    )
    model = Sequential([
        Input(shape=input_shape),
        LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(4e-4)),  
        Dropout(0.3),
        LSTM(88, return_sequences=False, kernel_regularizer=regularizers.l1(2e-4)), 
        Dropout(0.2),
        Flatten(),
        Dense(96, activation='relu', kernel_regularizer=regularizers.l2(5e-5)),
        Dense(1)  
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model