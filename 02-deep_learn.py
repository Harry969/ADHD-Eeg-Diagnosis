import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 加载和预处理数据
data_folder = r'E:\ADHD_EEG_Refinement\DATA\ASR'  # ASR文件夹路径
all_data = []

# 加载每个文件
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                print(f"警告: {file_name} 是空文件")
                continue
            
            # 只保留前20列（19个脑电通道 + label）
            columns_to_keep = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                               'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
            data = data[columns_to_keep]
            
            # 获取标签（A开头为ADHD，C开头为正常）
            label = 1 if file_name.startswith('A') else 0
            data['label'] = label
            
            # 只保留没有NaN的行
            data = data.dropna()
            if not data.empty:
                all_data.append(data)
                print(f"成功加载文件: {file_name}, 形状: {data.shape}")
            else:
                print(f"警告: {file_name} 清理NaN后为空")
                
        except Exception as e:
            print(f"加载文件 {file_name} 时出错: {str(e)}")

# 检查是否成功加载了数据
if not all_data:
    raise ValueError("没有成功加载任何数据文件！")

# 合并所有数据
df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后数据形状: {df.shape}")

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

print(f"\n特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\n训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# 3. 构建模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # 输入层 + 第一层
    Dropout(0.3),  # Dropout层
    Dense(64, activation='relu'),  # 第二层
    Dropout(0.3),  # Dropout层
    Dense(32, activation='relu'),  # 第三层
    Dropout(0.3),  # Dropout层
    Dense(1, activation='sigmoid')  # 输出层，用sigmoid处理二分类
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),  # 使用Adam优化器，学习率设置为0.001
    loss='binary_crossentropy',  # 二分类任务的损失函数
    metrics=['accuracy']  # 评估指标是准确率
)

# 4. 添加早停
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 5. 训练模型
history = model.fit(
    X_train, 
    y_train,
    epochs=50,  # 训练50轮
    batch_size=16,  # 每个批次使用16个样本
    validation_split=0.2,  # 使用20%的数据作为验证集
    callbacks=[early_stopping]  # 使用EarlyStopping防止过拟合
)

# 6. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# 7. 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 保存模型
model.save('adhd_classification_model2.h5')


# 代码2：数据加载与处理
# 特点：

# 保留了 所有 19 个脑电通道，并对数据进行清理（删除 NaN 值）。
# dropna() 用于清除带有 NaN 值的数据行，确保数据的完整性。
# 数据没有进行特征工程，而是直接用每个脑电通道的原始信号训练模型。
# 优点：

# 较完整的原始数据：保留了所有通道的原始数据，不会丢失任何通道的信息，可能更适合复杂的分类任务。
# 灵活性：不做任何特征选择或提取，保留原始信号数据，之后可以使用更复杂的特征提取或深度学习方法。
# 缺点：

# 内存和计算需求高：保留所有通道的数据会增加计算复杂性和内存需求，特别是在数据量较大时。
# 数据冗余：直接使用原始信号进行训练可能会导致训练时间较长，且需要较多的计算资源。
# 适用情况：

# 如果你的目标是 高精度 分类，并且数据量较小或计算能力足够强，保留原始信号并使用深度学习模型（如CNN或LSTM）可能会带来更好的效果。
# 如果你打算使用复杂的特征提取或深度学习模型，这个代码段会更适合。