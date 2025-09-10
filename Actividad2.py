import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# ============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================================================

print("=" * 60)
print("📊 DETECCIÓN DE SPAM CON REGRESIÓN LOGÍSTICA")
print("=" * 60)

# Cargar el archivo CSV
df = pd.read_csv("C:\\Users\\braro\\OneDrive\\Escritorio\\UNIVERSIDAD\\Machine Learning\\Actividad 1\\email_dataset.csv")

# Mezclar los datos para evitar sesgos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("🔀 Datos mezclados aleatoriamente")

# Exploración básica del dataset
print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"• Total de emails: {len(df)}")
print(f"• Emails SPAM: {df['label'].value_counts()['SPAM']} ({df['label'].value_counts()['SPAM']/len(df)*100:.1f}%)")
print(f"• Emails HAM: {df['label'].value_counts()['HAM']} ({df['label'].value_counts()['HAM']/len(df)*100:.1f}%)")

print(f"\n🔍 PRIMERAS 3 FILAS:")
print(df[['email_text', 'label']].head(5))

# ============================================================================
# 2. PREPROCESAMIENTO DEL TEXTO
# ============================================================================

print("\n" + "=" * 60)
print("🔧 PREPROCESAMIENTO DEL TEXTO")
print("=" * 60)

# Descargar stopwords en español
print("📥 Configurando stopwords...")
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('spanish'))
    print(f"✅ {len(stop_words)} stopwords cargadas")
except:
    print("⚠️  Error con stopwords, continuando sin ellas")
    stop_words = set()

def limpiar_texto(texto):
    """
    Limpia el texto eliminando puntuación, convirtiendo a minúsculas
    y removiendo stopwords
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar signos de puntuación y caracteres especiales
    texto = re.sub(r'\W', ' ', texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto)
    
    # Remover stopwords
    texto = ' '.join([palabra for palabra in texto.split() if palabra not in stop_words])
    
    return texto.strip()

# Aplicar limpieza al texto
print("�� Limpiando texto...")
df['texto_limpio'] = df['email_text'].apply(limpiar_texto)

print("✅ Texto limpiado")
print(f"📝 Ejemplo original: {df['email_text'].iloc[0][:80]}...")
print(f"📝 Ejemplo limpio: {df['texto_limpio'].iloc[0][:80]}...")

# ============================================================================
# 3. VECTORIZACIÓN CON TF-IDF
# ============================================================================

print("\n" + "=" * 60)
print("🔢 VECTORIZACIÓN DEL TEXTO")
print("=" * 60)

print("�� Convirtiendo texto a números con TF-IDF...")

# Crear vectorizador TF-IDF
vectorizador = TfidfVectorizer(
    max_features=1000,  # 1000 palabras más importantes
    ngram_range=(1, 2),  # Unigramas y bigramas
    min_df=2,  # Palabra debe aparecer en al menos 2 documentos
    max_df=0.95  # Palabra no debe aparecer en más del 95% de documentos
)

# Vectorizar el texto limpio
X = vectorizador.fit_transform(df['texto_limpio']).toarray()
y = df['label']

print(f"✅ Vectorización completada")
print(f"📊 Dimensiones: {X.shape[0]} emails × {X.shape[1]} características")

# ============================================================================
# 4. DIVISIÓN DE DATOS
# ============================================================================

print("\n" + "=" * 60)
print("📊 DIVISIÓN DE DATOS")
print("=" * 60)

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantener proporción de clases
)

print(f"✅ División completada:")
print(f"• Entrenamiento: {X_train.shape[0]} emails")
print(f"• Prueba: {X_test.shape[0]} emails")

# ============================================================================
# 5. ENTRENAMIENTO DEL MODELO
# ============================================================================

print("\n" + "=" * 60)
print("🤖 ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA")
print("=" * 60)

print("�� ¿Por qué Regresión Logística?")
print("• Perfecta para clasificación binaria (SPAM vs HAM)")
print("• Proporciona probabilidades interpretables")
print("• Robusta y funciona excelente con texto vectorizado")
print("• Es el estándar en Machine Learning para clasificación")

# Crear y entrenar modelo
print("\n🔄 Entrenando modelo...")
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_train, y_train)

print("✅ Modelo entrenado correctamente")

# Hacer predicciones
print("🔮 Realizando predicciones...")
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)

print("✅ Predicciones completadas")

# ============================================================================
# 6. EVALUACIÓN DEL MODELO
# ============================================================================

print("\n" + "=" * 60)
print("📈 EVALUACIÓN DEL MODELO")
print("=" * 60)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='SPAM')
recall = recall_score(y_test, y_pred, pos_label='SPAM')
f1 = f1_score(y_test, y_pred, pos_label='SPAM')

print(f"\n📊 MÉTRICAS PRINCIPALES:")
print(f"• Precisión general: {accuracy:.3f}")
print(f"• Precisión SPAM: {precision:.3f}")
print(f"• Sensibilidad SPAM: {recall:.3f}")
print(f"• F1-Score SPAM: {f1:.3f}")

# Reporte detallado
print(f"\n📊 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print(f"\n📊 MATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_test, y_pred)
print("                 Predicción")
print("                 HAM  SPAM")
print(f"Realidad HAM    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"         SPAM   {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================================================
# 7. INTERPRETACIÓN DE RESULTADOS
# ============================================================================

print("\n" + "=" * 60)
print("🔍 INTERPRETACIÓN DE RESULTADOS")
print("=" * 60)

print(f"\n�� ANÁLISIS DE RENDIMIENTO:")

# Análisis de Accuracy
if accuracy >= 0.95:
    print(f"✅ ACCURACY ({accuracy:.3f}): EXCELENTE - Más del 95% de emails clasificados correctamente")
elif accuracy >= 0.90:
    print(f"✅ ACCURACY ({accuracy:.3f}): MUY BUENO - Más del 90% de emails clasificados correctamente")
elif accuracy >= 0.80:
    print(f"⚠️  ACCURACY ({accuracy:.3f}): BUENO - Más del 80% de emails clasificados correctamente")
else:
    print(f"❌ ACCURACY ({accuracy:.3f}): REGULAR - El modelo necesita mejoras")

# Análisis de Precision
if precision >= 0.95:
    print(f"✅ PRECISION ({precision:.3f}): EXCELENTE - 95%+ de emails SPAM clasificados son realmente SPAM")
elif precision >= 0.90:
    print(f"✅ PRECISION ({precision:.3f}): MUY BUENO - 90%+ de emails SPAM clasificados son realmente SPAM")
else:
    print(f"⚠️  PRECISION ({precision:.3f}): BUENO - Algunos falsos positivos")

# Análisis de Recall
if recall >= 0.95:
    print(f"✅ RECALL ({recall:.3f}): EXCELENTE - Detecta 95%+ de todos los emails SPAM reales")
elif recall >= 0.90:
    print(f"✅ RECALL ({recall:.3f}): MUY BUENO - Detecta 90%+ de todos los emails SPAM reales")
else:
    print(f"⚠️  RECALL ({recall:.3f}): BUENO - Se pierde algunos emails SPAM")

# ============================================================================
# 8. ANÁLISIS DE PALABRAS IMPORTANTES
# ============================================================================

print("\n" + "=" * 60)
print("⭐ ANÁLISIS DE PALABRAS IMPORTANTES")
print("=" * 60)

# Obtener palabras más importantes del modelo
importancia = np.mean(X, axis=0)
palabras = vectorizador.get_feature_names_out()
top_features = sorted(zip(importancia, palabras), reverse=True)[:15]

print(f"\n�� TOP 15 PALABRAS MÁS IMPORTANTES:")
for i, (peso, palabra) in enumerate(top_features, 1):
    print(f"   {i:2d}. {palabra:20s}: {peso:.4f}")

# Analizar palabras por tipo (SPAM vs HAM)
spam_emails = df[df['label'] == 'SPAM']['texto_limpio']
ham_emails = df[df['label'] == 'HAM']['texto_limpio']

spam_words = ' '.join(spam_emails).split()
ham_words = ' '.join(ham_emails).split()

spam_word_freq = Counter(spam_words)
ham_word_freq = Counter(ham_words)

print(f"\n�� TOP 10 PALABRAS MÁS COMUNES EN SPAM:")
for palabra, freq in spam_word_freq.most_common(10):
    print(f"   • {palabra}: {freq} veces")

print(f"\n�� TOP 10 PALABRAS MÁS COMUNES EN HAM:")
for palabra, freq in ham_word_freq.most_common(10):
    print(f"   • {palabra}: {freq} veces")

# Analizar indicadores de SPAM vs HAM
spam_indicators = []
ham_indicators = []

for peso, palabra in top_features[:10]:
    spam_count = spam_word_freq.get(palabra, 0)
    ham_count = ham_word_freq.get(palabra, 0)
    
    if spam_count > ham_count:
        spam_indicators.append((palabra, peso, spam_count, ham_count))
    else:
        ham_indicators.append((palabra, peso, spam_count, ham_count))

print(f"\n🔴 INDICADORES DE SPAM (aparecen más en emails SPAM):")
for palabra, peso, spam_count, ham_count in spam_indicators[:5]:
    print(f"   • {palabra}: peso={peso:.4f}, SPAM={spam_count}, HAM={ham_count}")

print(f"\n�� INDICADORES DE HAM (aparecen más en emails legítimos):")
for palabra, peso, spam_count, ham_count in ham_indicators[:5]:
    print(f"   • {palabra}: peso={peso:.4f}, SPAM={spam_count}, HAM={ham_count}")

# ============================================================================
# 9. EJEMPLOS DE PREDICCIONES
# ============================================================================

print("\n" + "=" * 60)
print("🔮 EJEMPLOS DE PREDICCIONES")
print("=" * 60)

print(f"\n📧 PRIMEROS 5 EMAILS DE PRUEBA:")
for i in range(min(5, len(y_test))):
    prob_spam = y_prob[i][1] if len(y_prob[i]) > 1 else y_prob[i][0]
    prediccion = "SPAM" if prob_spam > 0.5 else "HAM"
    correcto = "✅" if prediccion == y_test.iloc[i] else "❌"
    print(f"   Email {i+1}: {prediccion} (prob: {prob_spam:.3f}) - Real: {y_test.iloc[i]} {correcto}")

print("\n" + "=" * 60)
print("🎉 ANÁLISIS COMPLETO FINALIZADO")
print("=" * 60)