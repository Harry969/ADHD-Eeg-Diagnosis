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
data_folder = r'E:\ADHD_EEG_Refinement\DATA\ASR'
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
            
            # 获取标签
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
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
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
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping]
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
model.save('adhd_classification_model.h5') 

# 代码1：特征提取和标签处理
# 特点：

# 在特征提取中，你只保留了部分通道（Fp1, Fp2, F3, F4, C3, C4, P3, P4），并且提取了每个通道的统计特征（均值、标准差、最大值、最小值）。
# 对每个文件提取的特征都是手动计算的统计量，作为模型的输入特征。
# 标签（ADHD 或正常对照组）是通过文件名来确定的。
# 优点：

# 特征简化：减少了输入特征的数量，只保留了部分关键通道并提取了简化的统计特征，减少了计算量。
# 较少的内存消耗：使用较少的通道和基本的统计特征，适合内存有限或计算能力有限的情况。
# 缺点：

# 信息损失：仅使用统计特征（均值、标准差等）可能会丢失信号的其他重要模式信息，比如时域或频域特征。
# 过于简化：如果EEG信号的模式较为复杂，可能不够细致。使用复杂的特征提取方法（如频谱分析、时序分析等）会更有优势。
# 适用情况：

# 如果你需要一个 快速、简洁 的模型，且数据特征不需要非常复杂的提取，使用这个代码段会更合适。