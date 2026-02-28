import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CustomerSegmenter:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X_processed = None
        self.preprocessor = None
        self.cluster_results = {}  # n_clusters -> (X_pca, labels, centers_pca)
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # o el número de núcleos que tú quieras

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

    def preprocess(self):
        print("Iniciando preprocesamiento...")
        
        features = ['gender', 'age', 'category', 'quantity', 'total_profit']
        df_filtered = self.df[features].dropna()

        numeric_features = ['age', 'quantity', 'total_profit']
        categorical_features = ['gender', 'category']

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

        self.X_processed = self.preprocessor.fit_transform(df_filtered)
        
        # Obtener los nombres de las columnas transformadas
        encoded_cat_columns = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_columns = numeric_features + list(encoded_cat_columns)

        # Convertir a DataFrame para visualizarlo
        df_transformed = pd.DataFrame(self.X_processed, columns=all_columns)
    
        print("Preprocesamiento completado. Primeras filas del resultado:")
        print(df_transformed.head())  # 👈 Aquí ves las primeras filas

    def fit_kmeans_and_reduce(self, cluster_range=(3, 6)):
        for n_clusters in range(*cluster_range):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_processed)
            centers = kmeans.cluster_centers_

            pca = PCA(n_components=2)
            X_array = self.X_processed.toarray() if hasattr(self.X_processed, "toarray") else self.X_processed
            X_pca = pca.fit_transform(X_array)
            centers_pca = pca.transform(centers)

            self.cluster_results[n_clusters] = (X_pca, labels, centers_pca)
        print("K-Means y reducción de dimensionalidad completados.")
        

    def plot_clusters_vertical(self):
        num_plots = len(self.cluster_results)
        fig, axes = plt.subplots(num_plots, 1, figsize=(9, 5 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for idx, n_clusters in enumerate(sorted(self.cluster_results)):
            X_pca, labels, centers_pca = self.cluster_results[n_clusters]
            ax = axes[idx]
            scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', ax=ax, legend=True, s=100)

            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, marker='X', label='Centro')

            cluster_labels = self.interpret_clusters(n_clusters)
            if not cluster_labels:
                continue

            for cluster_id, desc in cluster_labels.items():
                cx, cy = centers_pca[cluster_id]
                ax.text(cx + 0.3, cy, f"{desc}", fontsize=9, weight='bold', color='black')

            ax.set_title(f'K-Means con {n_clusters} Clusters')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')

            # Agregar el número de clusters en la parte superior derecha
            ax.text(0.98, 0.98, f'Clusters: {n_clusters}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=12, weight='bold', color='black')

            # Activar la leyenda
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
        print("Gráficas generadas con centros y descripciones.")
        
        # Guardar el gráfico como PNG
        plt.tight_layout()
        plt.savefig(f"clusters_{n_clusters}_png.png", format="png")  # Guarda el gráfico
        plt.show()  # Muestra el gráfico
        print("Gráficas generadas con centros y descripciones.")

        
    def interpret_clusters(self, n_clusters):
        """Imprime estadísticas descriptivas para interpretar los grupos"""
        if n_clusters not in self.cluster_results:
            print(f"No se encontraron resultados para {n_clusters} clusters.")
            return

        _, labels, _ = self.cluster_results[n_clusters]

        # Recuperar los datos originales filtrados
        features = ['gender', 'age', 'category', 'quantity', 'total_profit']
        df_filtered = self.df[features].dropna().copy()
        df_filtered['cluster'] = labels

        print(f"\n📊 Interpretación de clusters (K={n_clusters})")

        # Estadísticas por cluster
        cluster_summary = df_filtered.groupby('cluster').agg({
            'age': ['mean', 'median'],
            'gender': lambda x: x.mode()[0],
            'category': lambda x: x.mode()[0],
            'quantity': ['mean', 'sum'],
            'total_profit': ['mean', 'sum']
        })
        print(cluster_summary)

        # 👉 Imprimir cantidad de clientes por cluster
        print(f"\n👥 Cantidad de clientes por cluster (K={n_clusters}):")
        print(df_filtered['cluster'].value_counts().sort_index())

        # 👉 Estadísticas detalladas por cluster
        print(f"\n📈 Estadísticas detalladas por cluster (describe) (K={n_clusters}):")
        print(df_filtered.groupby('cluster').describe().transpose())
        
        print(df_filtered.groupby('cluster').describe().transpose())

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 filas, 1 columna

        

    def clasificacion_svm(self):
        print("\n🤖 Clasificación del nivel de gasto con SVM")

        df = self.df.copy()
        df['gasto_nivel'] = pd.qcut(df['total_profit'], q=3, labels=['bajo', 'medio', 'alto'])

        X = df[['age', 'quantity']]
        y = df['gasto_nivel']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Agregamos class_weight='balanced'
        model = SVC(kernel='linear', class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("Distribución de clases en el conjunto de entrenamiento:")
        print(y_train.value_counts())
        
        cm = confusion_matrix(y_test, y_pred, labels=['alto', 'bajo', 'medio'])

        # Mostrar como texto
        print(cm)

        # Mostrar como gráfico
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['alto', 'bajo', 'medio'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Matriz de Confusión')
        plt.show()
        
# --- USO DEL SCRIPT ---

if __name__ == "__main__":
    segmenter = CustomerSegmenter("DataBases/0. Different_stores_data_V2.csv")
    segmenter.load_data()
    segmenter.preprocess()
    segmenter.fit_kmeans_and_reduce(cluster_range=(3, 6))
    segmenter.interpret_clusters(n_clusters=3)
    segmenter.interpret_clusters(n_clusters=4)
    segmenter.interpret_clusters(n_clusters=5)
    segmenter.clasificacion_svm()
    segmenter.plot_clusters_vertical()