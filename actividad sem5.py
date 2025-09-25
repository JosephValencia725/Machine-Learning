#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador SPAM/HAM con Árbol de Decisión (CART) usando scikit-learn.
Autoría: (rellenar con los integrantes)
Repositorio: (agregar URL de GitHub)

Uso:
    python spam_ham_cart.py --data email_dataset.csv --runs 60

Entradas:
    --data : Ruta al CSV del dataset (usar el conjunto de la Actividad 1).
    --runs : Número de ejecuciones (>= 50). Por defecto 60.

Salidas (en el directorio de trabajo):
    - experiments_results.csv           -> Resultados por ejecución.
    - accuracy_by_run.png               -> Gráfica de Exactitud por ejecución.
    - f1_by_run.png                     -> Gráfica de F1-score por ejecución.
    - z_by_run.png                      -> Gráfica de Z-score (exactitud vs 50%).
    - Informe_SPAM_HAM_CART.pdf         -> Reporte PDF con resumen, gráficas y conclusiones.

Descripción del procedimiento:
    1) Limpieza básica del texto.
    2) Vectorización TF-IDF con 1–2-gramas.
    3) Modelo Árbol de Decisión (CART, criterio Gini) con distintas profundidades.
    4) Repetición >= 50 veces variando semilla, tamaño de test y profundidad.
    5) Métricas: Exactitud, F1-score y Z-score (exactitud vs 50%).
    6) Se generan gráficas y un informe PDF con conclusiones automáticas.

Requisitos:
    - Python 3.8+
    - scikit-learn, matplotlib, pandas, numpy
        pip install scikit-learn matplotlib pandas numpy
"""

import argparse
import re
from pathlib import Path
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Etiquetas consideradas como clase positiva (spam) si el dataset no está normalizado
POSITIVE_LABEL_NAMES = {"spam", "spams", "correo no deseado", "no_deseado", "no-deseado"}


# -------------------- Utilidades --------------------
def limpiar_texto(s: str) -> str:
    """Limpieza muy básica del texto."""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)                # URLs
    s = re.sub(r"[^a-záéíóúñü0-9\s@._-]+", " ", s)              # quitar símbolos raros
    s = re.sub(r"\s+", " ", s).strip()                           # espacios extra
    return s


def detectar_columnas(df: pd.DataFrame):
    """
    Heurística para detectar columna de texto y de etiqueta.
    - Etiqueta: baja cardinalidad (<=10) y que contenga 'spam/ham' o '0/1' u otros nombres frecuentes.
    - Texto: nombre típico (text, message, mensaje, email, content, correo, body) o columna con mayor longitud media de string.
    """
    cols = list(df.columns)

    # Candidatas a etiqueta por baja cardinalidad
    label_cands = []
    for c in cols:
        uniq = df[c].dropna().unique()
        if len(uniq) <= 10:
            low = set(map(lambda x: str(x).strip().lower(), uniq))
            if {"spam", "ham"} & low or {"0", "1"} & low or low & POSITIVE_LABEL_NAMES:
                label_cands.append(c)
    if not label_cands:
        # fallback: última columna de baja cardinalidad
        for c in reversed(cols):
            uniq = df[c].dropna().unique()
            if len(uniq) <= 10:
                label_cands.append(c)
                break

    # Candidatas a texto por nombre
    name_pref = ["text", "message", "mensaje", "email", "content", "correo", "body"]
    text_cands_named = [c for c in cols if any(n in c.lower() for n in name_pref)]
    # Fallback: mayor longitud media
    avg_len = {c: df[c].astype(str).str.len().mean() for c in cols}
    text_cand_by_len = max(avg_len, key=avg_len.get) if avg_len else cols[0]

    text_col = text_cands_named[0] if text_cands_named else text_cand_by_len
    label_col = label_cands[0] if label_cands else (cols[-1] if len(cols) > 1 else cols[0])
    return text_col, label_col


def crear_etiquetas_binarias(y_raw: pd.Series):
    """
    Mapea diferentes formatos de etiqueta a {0: ham, 1: spam}.
    Si no hay nombres claros, aplica una heurística (clase minoritaria = spam).
    """
    y_str = y_raw.astype(str).str.strip().str.lower()
    if set(y_str.unique()) & {"spam", "ham"}:
        y = y_str.map(lambda v: 1 if v == "spam" else 0)
        pos_label = 1
    else:
        y = y_str.map(lambda v: 1 if (v in POSITIVE_LABEL_NAMES or v == "spam" or v == "1")
                      else (0 if v in {"ham", "0"} else np.nan))
        if y.isna().any():
            counts = Counter(y_str)
            ordenados = [lbl for lbl, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
            if len(ordenados) >= 2:
                ham_like, spam_like = ordenados[0], ordenados[1]
                y = y_str.map(lambda v: 1 if v == spam_like else (0 if v == ham_like else np.nan))
            else:
                # Último recurso: binarizar determinísticamente
                unicos = sorted(y_str.unique())
                mapping = {unicos[0]: 0}
                if len(unicos) > 1:
                    mapping[unicos[1]] = 1
                y = y_str.map(mapping).astype(float)
        # No forzar a int aquí; permitir NaN y filtrar más adelante en main
        pos_label = 1
    return y, pos_label


def z_score_binomial(exactitud: float, n: int, p0: float = 0.5) -> float:
    """
    Z-score de la proporción de aciertos frente a una línea base p0 (por defecto 0.5).
    Aproximación normal.
    """
    if n == 0 or p0 * (1 - p0) == 0:
        return float("nan")
    return (exactitud - p0) / np.sqrt(p0 * (1 - p0) / n)


# -------------------- Programa principal --------------------
def main():
    parser = argparse.ArgumentParser(description="Clasificador SPAM/HAM con CART (scikit-learn)")
    parser.add_argument("--data", type=str, default=None, help="Ruta al CSV con el dataset de correos")
    parser.add_argument("--runs", type=int, default=60, help="Número de ejecuciones (>=50). Por defecto 60")
    args = parser.parse_args()

    # Resolver ruta del CSV con varias opciones por defecto
    posibles_rutas = []
    if args.data:
        posibles_rutas.append(Path(args.data))
    try:
        posibles_rutas.append(Path(__file__).with_name("email_dataset.csv"))
    except NameError:
        pass  # __file__ puede no existir en algunos entornos interactivos
    posibles_rutas.append(Path.cwd() / "email_dataset.csv")

    data_path = None
    for cand in posibles_rutas:
        if cand is not None and cand.exists():
            data_path = cand
            break
    if data_path is None:
        sugerencia = "; ".join(str(p) for p in posibles_rutas)
        raise FileNotFoundError(
            "No se encontró el CSV. Pasa --data o coloca 'email_dataset.csv' en una de estas rutas: " + sugerencia
        )

    # Leer datos con mayor robustez de codificación
    df = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        # Intento final sin especificar encoding
        df = pd.read_csv(data_path)

    # Detectar columnas
    text_col, label_col = detectar_columnas(df)
    X_text = df[text_col].fillna("").astype(str).map(limpiar_texto)
    y, pos_label = crear_etiquetas_binarias(df[label_col])
    # Filtrar filas con etiquetas inválidas (NaN) y alinear X e y
    mask_valid = ~pd.isna(y)
    X_text = X_text[mask_valid].reset_index(drop=True)
    y = y[mask_valid].astype(int).reset_index(drop=True)

    # Configuración de experimentos
    N_RUNS = max(50, args.runs)  # asegurar al menos 50
    RANDOM_SEEDS = list(range(1, N_RUNS + 1))
    TEST_SIZES = [0.2, 0.25, 0.3]
    MAX_DEPTHS = [None, 10, 20, 30]  # controlar sesgo/varianza

    # Generar combinaciones (semilla, tamaño de test, profundidad)
    combos = list(product(RANDOM_SEEDS, TEST_SIZES, MAX_DEPTHS))
    if len(combos) > N_RUNS:
        combos = combos[:N_RUNS]  # tomar primeras N combinaciones

    resultados = []
    for run_id, (seed, test_size, max_depth) in enumerate(combos, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=seed,
            stratify=y if len(np.unique(y)) == 2 else None
        )

        # Pipeline: TF-IDF (1–2-gramas) + Árbol CART (Gini)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("cart", DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=seed)),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        z_acc = z_score_binomial(acc, n=len(y_test), p0=0.5)

        resultados.append({
            "run": run_id,
            "seed": seed,
            "test_size": test_size,
            "max_depth": -1 if max_depth is None else int(max_depth),
            "n_test": int(len(y_test)),
            "accuracy": float(acc),
            "f1_score": float(f1),
            "z_score_accuracy_vs_50pct": float(z_acc),
        })

    df_res = pd.DataFrame(resultados)

    # Z-score estandarizado (para visualizar variaciones entre ejecuciones)
    for col in ["accuracy", "f1_score", "z_score_accuracy_vs_50pct"]:
        mu = df_res[col].mean()
        sigma = df_res[col].std(ddof=1)
        df_res[f"{col}_z_standardized"] = (df_res[col] - mu) / sigma if sigma > 0 else np.nan

    # Guardar CSV con resultados
    out_csv = Path("experiments_results.csv")
    df_res.to_csv(out_csv, index=False)

    # --------- Gráficas ---------
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_res["run"], df_res["accuracy"], marker="o", linestyle="-")
    plt.xlabel("Ejecución")
    plt.ylabel("Exactitud")
    plt.title("Exactitud por ejecución (CART)")
    plt.tight_layout()
    plt.savefig("accuracy_by_run.png")
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df_res["run"], df_res["f1_score"], marker="o", linestyle="-")
    plt.xlabel("Ejecución")
    plt.ylabel("F1-score")
    plt.title("F1-score por ejecución (CART)")
    plt.tight_layout()
    plt.savefig("f1_by_run.png")
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df_res["run"], df_res["z_score_accuracy_vs_50pct"], marker="o", linestyle="-")
    plt.xlabel("Ejecución")
    plt.ylabel("Z (vs 50%)")
    plt.title("Z-score de Exactitud vs 50% por ejecución")
    plt.tight_layout()
    plt.savefig("z_by_run.png")
    plt.close()

    # --------- Informe PDF ---------
    acc_mean, acc_std = df_res["accuracy"].mean(), df_res["accuracy"].std(ddof=1)
    f1_mean, f1_std = df_res["f1_score"].mean(), df_res["f1_score"].std(ddof=1)
    z_mean, z_std = df_res["z_score_accuracy_vs_50pct"].mean(), df_res["z_score_accuracy_vs_50pct"].std(ddof=1)

    mejor = df_res.sort_values(by=["f1_score", "accuracy"], ascending=False).iloc[0].to_dict()
    peor = df_res.sort_values(by=["f1_score", "accuracy"], ascending=True).iloc[0].to_dict()

    def describe_row(row):
        maxd = "None" if int(row["max_depth"]) == -1 else str(int(row["max_depth"]))
        return (f"- seed={int(row['seed'])}, test_size={row['test_size']}, "
                f"max_depth={maxd}, acc={row['accuracy']:.3f}, "
                f"f1={row['f1_score']:.3f}, z={row['z_score_accuracy_vs_50pct']:.2f}")

    conclusiones = f"""
Resumen de resultados (promedios ± std):
- Exactitud: {acc_mean:.3f} ± {acc_std:.3f}
- F1-score: {f1_mean:.3f} ± {f1_std:.3f}
- Z-score de exactitud vs 50%: {z_mean:.2f} ± {z_std:.2f}

Mejor configuración (por F1 y luego exactitud):
{describe_row(mejor)}

Peor configuración (por F1 y luego exactitud):
{describe_row(peor)}

Posibles explicaciones de la variabilidad observada:
1) Diferentes particiones de entrenamiento/test (semillas y tamaños de test) cambian qué ejemplos “difíciles” quedan en test.
2) La profundidad del árbol controla sesgo/varianza: mayor profundidad puede sobreajustar; menor, infraajustar.
3) Si existe desbalance de clases, F1 es más sensible que la exactitud: clasificar conservadoramente eleva exactitud pero puede bajar F1.
4) La representación TF-IDF (1–2-gramas) interactúa con el árbol y puede inducir alta varianza según la muestra.
"""

    pdf_path = Path("Informe_SPAM_HAM_CART.pdf")
    with PdfPages(pdf_path) as pdf:
        # Portada / resumen
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
        plt.axis("off")
        portada = (
            "Clasificador SPAM/HAM con CART (scikit-learn)\n\n"
            "Integrantes del grupo:\n"
            " - [Escribe aquí los nombres]\n\n"
            f"Dataset: {data_path.name}\n"
            f"Texto: '{text_col}' | Etiqueta: '{label_col}'\n\n"
            "Procedimiento (resumen):\n"
            "1) Limpieza de texto y vectorización TF-IDF (1–2-gramas).\n"
            "2) Modelo DecisionTreeClassifier (CART, criterio Gini).\n"
            "3) ≥50 ejecuciones variando semilla, tamaño de test y max_depth.\n"
            "4) Métricas: Exactitud, F1-score y Z-score (exactitud vs 50%).\n\n"
            "Resultados (promedios ± std):\n"
            f"- Exactitud: {acc_mean:.3f} ± {acc_std:.3f}\n"
            f"- F1-score: {f1_mean:.3f} ± {f1_std:.3f}\n"
            f"- Z-score (vs 50%): {z_mean:.2f} ± {z_std:.2f}\n\n"
            "Ver la última página para conclusiones y explicación de variaciones."
        )
        plt.text(0.05, 0.95, portada, va="top", ha="left", wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # Gráficas
        for img, title in [("accuracy_by_run.png", "Exactitud por ejecución"),
                           ("f1_by_run.png", "F1-score por ejecución"),
                           ("z_by_run.png", "Z-score de Exactitud vs 50% por ejecución")]:
            try:
                im = plt.imread(img)
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(im); plt.axis("off"); plt.title(title)
                pdf.savefig(fig); plt.close(fig)
            except Exception:
                pass

        # Conclusiones
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.05, 0.95, conclusiones, va="top", ha="left", wrap=True)
        pdf.savefig(fig); plt.close(fig)

    print("Listo. Archivos generados:")
    print(f"- {out_csv.resolve()}")
    print(f"- accuracy_by_run.png, f1_by_run.png, z_by_run.png")
    print(f"- {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
