import argparse
import re
from pathlib import Path
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

POSITIVE_LABEL_NAMES = {"spam", "spams", "correo no deseado", "no_deseado", "no-deseado"}


def limpiar_texto(s: str) -> str:
    """Limpieza básica de texto."""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"[^a-záéíóúñü0-9\s@._-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detectar_columnas(df: pd.DataFrame):
    """Detecta columnas de texto y etiqueta usando heurísticas simples."""
    cols = list(df.columns)

    label_cands = []
    for c in cols:
        uniq = df[c].dropna().unique()
        if len(uniq) <= 10:
            low = set(map(lambda x: str(x).strip().lower(), uniq))
            if {"spam", "ham"} & low or {"0", "1"} & low or low & POSITIVE_LABEL_NAMES:
                label_cands.append(c)
    if not label_cands:
        for c in reversed(cols):
            uniq = df[c].dropna().unique()
            if len(uniq) <= 10:
                label_cands.append(c)
                break

    name_pref = ["text", "message", "mensaje", "email", "content", "correo", "body"]
    text_cands_named = [c for c in cols if any(n in c.lower() for n in name_pref)]
    avg_len = {c: df[c].astype(str).str.len().mean() for c in cols}
    text_cand_by_len = max(avg_len, key=avg_len.get) if avg_len else cols[0]

    text_col = text_cands_named[0] if text_cands_named else text_cand_by_len
    label_col = label_cands[0] if label_cands else (cols[-1] if len(cols) > 1 else cols[0])
    return text_col, label_col


def crear_etiquetas_binarias(y_raw: pd.Series):
    """Mapea etiquetas a {0: ham, 1: spam}. Mantiene NaN para valores desconocidos."""
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
                unicos = sorted(y_str.unique())
                mapping = {unicos[0]: 0}
                if len(unicos) > 1:
                    mapping[unicos[1]] = 1
                y = y_str.map(mapping).astype(float)
        pos_label = 1
    return y, pos_label


def z_score_binomial(exactitud: float, n: int, p0: float = 0.5) -> float:
    """Z-score de proporción frente a p0 usando aproximación normal."""
    if n == 0 or p0 * (1 - p0) == 0:
        return float("nan")
    return (exactitud - p0) / np.sqrt(p0 * (1 - p0) / n)


def main():
    parser = argparse.ArgumentParser(description="Clasificador SPAM/HAM con CART (scikit-learn)")
    parser.add_argument("--data", type=str, default=None, help="Ruta al CSV con el dataset de correos")
    parser.add_argument("--runs", type=int, default=60, help="Número de ejecuciones (>=50). Por defecto 60")
    args = parser.parse_args()

    # Resolver ruta del CSV (argumento, misma carpeta del script, o cwd)
    posibles_rutas = []
    if args.data:
        posibles_rutas.append(Path(args.data))
    try:
        posibles_rutas.append(Path(__file__).with_name("email_dataset.csv"))
    except NameError:
        pass
    posibles_rutas.append(Path.cwd() / "email_dataset.csv")

    data_path = None
    for cand in posibles_rutas:
        if cand is not None and cand.exists():
            data_path = cand
            break
    if data_path is None:
        sugerencia = "; ".join(str(p) for p in posibles_rutas)
        raise FileNotFoundError(
            "No se encontró el CSV. Pasa --data o coloca 'email_dataset.csv' en: " + sugerencia
        )

    # Leer CSV
    df = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(data_path)

    # Columnas, limpieza y etiquetas
    text_col, label_col = detectar_columnas(df)
    X_text = df[text_col].fillna("").astype(str).map(limpiar_texto)
    y, pos_label = crear_etiquetas_binarias(df[label_col])
    mask_valid = ~pd.isna(y)
    X_text = X_text[mask_valid].reset_index(drop=True)
    y = y[mask_valid].astype(int).reset_index(drop=True)

    # Configuración de experimentos
    N_RUNS = max(50, args.runs)
    RANDOM_SEEDS = list(range(1, N_RUNS + 1))
    TEST_SIZES = [0.2, 0.25, 0.3]
    MAX_DEPTHS = [None, 10, 20, 30]

    combos = list(product(RANDOM_SEEDS, TEST_SIZES, MAX_DEPTHS))
    if len(combos) > N_RUNS:
        combos = combos[:N_RUNS]

    resultados = []
    for run_id, (seed, test_size, max_depth) in enumerate(combos, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=seed,
            stratify=y if len(np.unique(y)) == 2 else None
        )

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

    # Estandarización para visualizar
    for col in ["accuracy", "f1_score", "z_score_accuracy_vs_50pct"]:
        mu = df_res[col].mean()
        sigma = df_res[col].std(ddof=1)
        df_res[f"{col}_z_standardized"] = (df_res[col] - mu) / sigma if sigma > 0 else np.nan

    # Gráficas
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

    print("Listo. Archivos generados:")
    print("- accuracy_by_run.png, f1_by_run.png, z_by_run.png")


if __name__ == "__main__":
    main()


