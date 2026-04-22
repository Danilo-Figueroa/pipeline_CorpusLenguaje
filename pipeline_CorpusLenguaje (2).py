# =====================================================
# PROCESAMIENTO DE LENGUAJE NATURAL - INFORME DE CORPUS
# =====================================================
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,PlaintextCorpusReader
import os
import string
import re  # Funciona para extraer solo el texto entre comillas
from collections import Counter 
#============
##FUNCIONES##
#============
def quitarStopwords_eng(texto):
    #Elimina las palabras vacías y signos de puntuación
    ingles = stopwords.words("english")
    return [w.lower() for w in texto if w.lower() not in ingles 
            and w not in string.punctuation 
            and w not in ["'s", '|', '--', "''", "``", ".-"]]

def get_wordnet_pos(word):
    #Mapea las etiquetas POS para una lematización precisa
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    #palabras a su forma base o lema
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]

####Inicialización del lematizador####
lemmatizer = WordNetLemmatizer()

###LECTURA Y EXTRACCIÓN DE DATOS####

# Definimos la ubicación del archivo
ruta_mia = r'C:/Windows' 
nombre_archivo = 'CorpusLenguajes.txt'

# Inicializa lector de NLTK
# Apunta al directorio  y filtra específicamente el archivo
lector_corpus = PlaintextCorpusReader(ruta_mia, nombre_archivo)

lineas_texto = []


###EXCEPCION##

try:
    # Obtenemos el texto bruto (raw) del archivo a través del lector
    contenido_bruto = lector_corpus.raw(nombre_archivo)
    
    # Procesamos línea por línea para extraer el contenido entre comillas
    for linea in contenido_bruto.splitlines():
        match = re.search(r'"([^"]*)"', linea)
        if match:
            lineas_texto.append(match.group(1))
            
except Exception as e:
    print(f"Error al procesar el corpus: {e}")
#################################
#    PROCESAMIENTO DEL CORPUS   #
#################################
corpus_procesado = []
for oracion in lineas_texto:
    # Aplicamos: Tokenizar -> Limpiar -> Lematizar
    tokens = word_tokenize(oracion)
    limpios = quitarStopwords_eng(tokens)
    lemas = lematizar(limpios)
    corpus_procesado.append(lemas)

###análisis TF-IDF###
corpus_final_texto = [' '.join(lista) for lista in corpus_procesado]

####CÁLCULO DE MATRIZ TF-IDF ####

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_final_texto)

print("\n--- MATRIZ TF-IDF ---")
print(X.toarray())

print("\n--- VOCABULARIO ---")
print(vectorizer.get_feature_names_out())

###########################################
#GENERACIÓN DEL INFORME DE CONCLUSIONES ---
###########################################


todas_las_palabras = [p for sub in corpus_procesado for p in sub]
conteo_global = Counter(todas_las_palabras)

print("\n" + "="*40)
print("       INFORME DE CONCLUSIONES")
print("="*40)

# 1. Jerarquía de las 6 palabras más usadas
print(f"1. Jerarquía (Top 6): {conteo_global.most_common(6)}")

# 2. Palabra menos utilizada
print(f"2. Palabra menos utilizada: {conteo_global.most_common()[-1]}")

# 3. Palabras más repetidas en la misma oración
print("\n3. Repeticiones internas por oración:")
for i, oracion_lemas in enumerate(corpus_procesado):
    repeticiones = [word for word, count in Counter(oracion_lemas).items() if count > 1]
    if repeticiones:
        print(f"   Oración {i+1}: {repeticiones}")

############################################
#GRAFICO DE DISTRIBUCIONN DE FRECUENCIA
############################################

# Las 6 palabras más frecuentes
palabras_graf, frecuencias_graf = zip(*conteo_global.most_common(6))

plt.figure(figsize=(10, 6))
plt.bar(palabras_graf, frecuencias_graf, color='steelblue')

plt.title("Distribución de Frecuencia (Lemas del Corpus)")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")

# evitar el cruce de palabras largas
plt.xticks(rotation=45, ha='right') 

plt.tight_layout()
plt.show()