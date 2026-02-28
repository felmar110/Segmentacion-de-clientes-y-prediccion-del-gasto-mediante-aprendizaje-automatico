import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping


class SpendingScorePredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_nn = None          # Para guardar la red neuronal
        self.model_lr = None          # Para guardar la regresión lineal
        self.preprocessor = None      # Para guardar el preprocesador de datos
        self.X_train_scaled = None    # Datos de entrenamiento escalados
        self.X_test_scaled = None     # Datos de prueba escalados
        self.X_train = None           # Datos de entrenamiento originales
        self.X_test = None            # Datos de prueba originales
        self.y_train = None           # Etiquetas de entrenamiento
        self.y_test = None            # Etiquetas de prueba
        self.y_pred_nn = None         # Predicciones de la red neuronal
        self.y_pred_lr = None         # Predicciones de la regresión lineal


    def load_and_prepare_data(self):
        df = pd.read_csv(self.file_path)
        df['spending_score'] = df['total_profit'] / df['quantity']
        X = df[['gender', 'age', 'category', 'quantity', 'total_profit']]
        y = df['spending_score']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        numeric_features = ['age', 'quantity', 'total_profit']
        categorical_features = ['gender', 'category']
        
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
        
        self.X_train_scaled = self.preprocessor.fit_transform(self.X_train)
        self.X_test_scaled = self.preprocessor.transform(self.X_test)

    def build_neural_network(self):
        self.model_nn = Sequential([
            Input(shape=(self.X_train_scaled.shape[1],)), 
            Dense(16, activation='relu'), # Rectified Linear Unit": función matemática para rapidez de aprendizaje (valores negativos cero y positivos iguales)
            Dense(8, activation='relu'),
            Dense(1)
        ])
        self.model_nn.compile(optimizer='adam', loss='mean_squared_error')
        
    def train_neural_network(self, epochs=100, batch_size=16):
        early_stop = EarlyStopping(
            monitor='val_loss',     # supervisa la pérdida en validación
            patience=10,            # si no mejora en 10 épocas, se detiene
            restore_best_weights=True
        )

        history = self.model_nn.fit(
            self.X_train_scaled,
            self.y_train,
            validation_split=0.2,   # usa el 20% de los datos de entrenamiento como validación
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0               # cambia a 1 si quieres ver progreso en consola
        )

        # Número real de épocas entrenadas
        actual_epochs = len(history.history['loss'])
        print(f"Entrenamiento detenido después de {actual_epochs} épocas.")

    #def train_neural_network(self, epochs=100, batch_size=16):
    #    self.model_nn.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate_neural_network(self):
        self.y_pred_nn = self.model_nn.predict(self.X_test_scaled).flatten()
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_nn))
        mae = mean_absolute_error(self.y_test, self.y_pred_nn)
        return rmse, mae

    def train_linear_regression(self):
        self.model_lr = Pipeline([
            ('preprocess', self.preprocessor),
            ('regressor', LinearRegression())
        ])
        self.model_lr.fit(self.X_train, self.y_train)

    def evaluate_linear_regression(self):
        self.y_pred_lr = self.model_lr.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_lr))
        mae = mean_absolute_error(self.y_test, self.y_pred_lr)
        return rmse, mae

    def plot_predictions(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Red Neuronal
        axes[0].scatter(self.y_test, self.y_pred_nn, alpha=0.6, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        axes[0].set_xlabel('Valor Real')
        axes[0].set_ylabel('Valor Predicho')
        axes[0].set_title('Red Neuronal: Real vs Predicho')
        axes[0].grid(True)

        # Regresión Lineal
        axes[1].scatter(self.y_test, self.y_pred_lr, alpha=0.6, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        axes[1].set_xlabel('Valor Real')
        axes[1].set_ylabel('Valor Predicho')
        axes[1].set_title('Regresión Lineal: Real vs Predicho')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_errors(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Errores Red Neuronal
        errors_nn = self.y_test - self.y_pred_nn
        axes[0].hist(errors_nn, bins=30, edgecolor='k', alpha=0.7, color='blue')
        axes[0].set_xlabel('Error de Predicción')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Errores - Red Neuronal')
        axes[0].grid(True)

        # Errores Regresión Lineal
        errors_lr = self.y_test - self.y_pred_lr
        axes[1].hist(errors_lr, bins=30, edgecolor='k', alpha=0.7, color='green')
        axes[1].set_xlabel('Error de Predicción')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Errores - Regresión Lineal')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        
    def plot_error_distribution_kde(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calcular errores
        errors_nn = self.y_test - self.y_pred_nn
        errors_lr = self.y_test - self.y_pred_lr

        # KDE Red Neuronal
        sns.kdeplot(errors_nn, label='Red Neuronal', color='blue', fill=True, alpha=0.5)

        # KDE Regresión Lineal
        sns.kdeplot(errors_lr, label='Regresión Lineal', color='green', fill=True, alpha=0.5)

        # Línea de referencia
        ax.axvline(0, color='red', linestyle='--')

        # Títulos y etiquetas
        ax.set_title('Distribución de Errores de Predicción (KDE)', fontsize=16)
        ax.set_xlabel('Error de Predicción (Valor Real - Valor Predicho)', fontsize=14)
        ax.set_ylabel('Densidad', fontsize=14)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

# Uso de la clase
if __name__ == "__main__":
    predictor = SpendingScorePredictor(file_path="DataBases/0. Different_stores_data_V2.csv")
    
    predictor.load_and_prepare_data()
    
    predictor.build_neural_network()
    predictor.train_neural_network()
    rmse_nn, mae_nn = predictor.evaluate_neural_network()
    
    predictor.train_linear_regression()
    rmse_lr, mae_lr = predictor.evaluate_linear_regression()
    
    print(f"Neural Network - RMSE: {rmse_nn:.2f}, MAE: {mae_nn:.2f}")
    print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
    
    predictor.plot_predictions()
    predictor.plot_errors()
    predictor.plot_error_distribution_kde()