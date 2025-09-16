import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def build_model() -> Pipeline:
    """
    Crea el flujo del modelo: primero escala los datos y luego aplica
    un clasificador lineal.
    """
    # Armamos el pipeline: escalado + modelo
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(random_state=42)),
        ]
    )
    return pipeline


def train_and_evaluate() -> dict:
    """
    Entrena y evalúa el modelo con validación cruzada y devuelve:
    modelo final, métricas y tablas útiles.
    """
    # Cargamos el dataset Iris (X: medidas de flores, y: tipo de flor)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Validación cruzada estratificada (5 partes) para medir precisión
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = build_model()
    y_pred_cv = cross_val_predict(base_model, X, y, cv=cv)

    # Precisión promedio con las predicciones de la validación
    acc = accuracy_score(y, y_pred_cv)

    # Entrenamos el modelo final con todos los datos
    final_model = build_model()
    final_model.fit(X, y)

    return {
        "model": final_model,
        "target_names": iris.target_names,
        "accuracy": acc,
        "report_dict": classification_report(y, y_pred_cv, target_names=iris.target_names, output_dict=True),
        "cm": confusion_matrix(y, y_pred_cv),
        "target_names": iris.target_names,
        "feature_names": iris.feature_names,
        "y_test": y,
        "y_pred": y_pred_cv,
    }


def predict_sample(model: Pipeline, sample: np.ndarray) -> int:
    """
    Predice la clase para una muestra de 4 características
    [longitud_sépalo, ancho_sépalo, longitud_pétalo, ancho_pétalo].
    Devuelve el índice de clase.
    """
    # Devuelve la clase como entero (0, 1 o 2)
    pred = model.predict(sample.reshape(1, -1))
    return int(pred[0])


def main(argv: list[str]) -> int:
    # Definimos argumentos de línea de comandos (opcional: --predict con 4 números)
    parser = argparse.ArgumentParser(
        description=(
            "Clasificador de Iris con scikit-learn (Regresión Lineal: RidgeClassifier).\n"
            "Si proporcionas --predict con 4 valores, predice la variedad."
        )
    )
    parser.add_argument(
        "--predict",
        nargs=4,
        type=float,
        metavar=("longitud_sépalo", "ancho_sépalo", "longitud_pétalo", "ancho_pétalo"),
        help="Medidas de la planta en el orden: longitud_sépalo ancho_sépalo longitud_pétalo ancho_pétalo",
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("🌸 CLASIFICADOR IRIS (Regresión Lineal - RidgeClassifier)")
    print("=" * 60)

    # Entrenamos y evaluamos el modelo
    results = train_and_evaluate()

    print("\n📈 Rendimiento en conjunto de prueba:")
    print(f"• Precisión: {results['accuracy']:.3f}")
    # Tabla: reporte de clasificación
    # Armamos una tabla con las métricas por clase
    report_df = pd.DataFrame(results["report_dict"]).T
    print("\n📊 Reporte de clasificación (tabla):")
    print(report_df.to_string(float_format=lambda v: f"{v:.3f}"))

    # Tabla: matriz de confusión con etiquetas
    # Matriz de confusión con nombres de clases
    cm_df = pd.DataFrame(
        results["cm"], index=results["target_names"], columns=results["target_names"]
    )
    print("\n📊 Matriz de confusión (tabla):")
    print(cm_df.to_string())

    # No hay matriz de confusión en regresión

    if args.predict is not None:
        # Si el usuario pasó 4 valores, hacemos una predicción con ellos
        sample = np.array([float(v) for v in args.predict], dtype=float)
        idx = predict_sample(results["model"], sample)
        clase = results["target_names"][idx]
        print("\n🔮 Predicción para la muestra ingresada:")
        print(f"• Clase predicha: {clase}")
        print(
            "• Notación de entrada [longitud_sépalo, ancho_sépalo, longitud_pétalo, ancho_pétalo]: "
            f"{sample.tolist()}"
        )
    else:
        # Ejemplo de uso si no se proporcionan valores
        ejemplo = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
        idx = predict_sample(results["model"], ejemplo)
        clase = results["target_names"][idx]
        print("\n🔎 Ejemplo de predicción (sin argumentos):")
        print("• Entrada de ejemplo: [5.1, 3.5, 1.4, 0.2]")
        print(f"• Clase predicha: {clase}")

    # Tabla: coeficientes del modelo (lineal) por clase y característica
    # Mostramos los coeficientes del modelo por clase y característica
    clf = results["model"].named_steps["clf"]
    coef_matrix = getattr(clf, "coef_", None)
    if coef_matrix is not None:
        coef_df = pd.DataFrame(
            coef_matrix,
            index=[f"class={name}" for name in results["target_names"]],
            columns=results["feature_names"],
        )
        print("\n🧮 Coeficientes del modelo (tabla):")
        print(coef_df.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n✔️ Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

