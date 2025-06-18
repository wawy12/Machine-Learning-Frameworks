import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Permute, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def r_squared(y_true, y_pred):
    """自定义R平方评估指标"""
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

def create_cnn_model(input_shape, initial_learning_rate, total_steps):
    """
    创建纯CNN模型架构
    
    参数:
    input_shape -- 输入数据的形状 (timesteps, features)
    initial_learning_rate -- 初始学习率
    total_steps -- 总训练步数（用于余弦衰减调度）
    
    返回:
    编译好的CNN模型
    """
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,  
        decay_steps=total_steps,  
        alpha=0.0  
    )
    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),  # 将特征和时间步重新排列
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
        Dense(1)  # 回归输出层
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model

def create_cnn_lstm_model(input_shape, initial_learning_rate, total_steps):
    """
    创建CNN-LSTM混合模型架构
    
    参数:
    input_shape -- 输入数据的形状 (timesteps, features)
    initial_learning_rate -- 初始学习率
    total_steps -- 总训练步数（用于余弦衰减调度）
    
    返回:
    编译好的CNN-LSTM模型
    """
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,  
        alpha=0.0  
    )
    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),  # 第一次排列：将特征维度放在时间步维度前
        Conv1D(48, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(3e-5)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        MaxPooling1D(pool_size=2),
        Conv1D(40, kernel_size=3, padding='same', kernel_regularizer=regularizers.l1(4e-4)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        MaxPooling1D(pool_size=2),
        Conv1D(48, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l1(6e-5)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        Permute((2, 1)),  # 第二次排列：恢复时间序列格式用于LSTM
        LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l1(2e-3)),
        Dropout(0.4),
        LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l1(5e-3)), 
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(2e-5)),
        Dense(1)  # 回归输出层
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model

def create_lstm_model(input_shape, initial_learning_rate, total_steps):
    """
    创建纯LSTM模型架构
    
    参数:
    input_shape -- 输入数据的形状 (timesteps, features)
    initial_learning_rate -- 初始学习率
    total_steps -- 总训练步数（用于余弦衰减调度）
    
    返回:
    编译好的LSTM模型
    """
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
        Dense(1)  # 回归输出层
    ])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', r_squared])
    return model