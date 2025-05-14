import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict

# Descargar recursos de NLTK (ejecutar solo una vez)
nltk.download('punkt')
nltk.download('stopwords')

def generate_summary(text, num_sentences=5):
    """
    Genera un resumen extractivo seleccionando las n frases más importantes
    basándose en la frecuencia de palabras clave.
    """
    # Tokenizar en oraciones
    sentences = sent_tokenize(text)
    
    # Tokenizar todas las palabras y eliminar stopwords
    stopwords_list = set(stopwords.words('spanish')) # Cambiar a 'english' si trabajas en inglés
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords_list]
    
    # Calcular frecuencia de palabras
    freq_table = FreqDist(words)
    
    # Calcular puntuación de cada oración
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[i] += freq_table[word]
    
    # Ordenar oraciones por puntuación y seleccionar las mejores
    best_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    best_sentences = sorted(best_sentences, key=lambda x: x[0])  # Ordenar por posición original
    
    # Crear el resumen
    summary = ' '.join([sentences[i] for i, _ in best_sentences])
    
    return summary

# Ejemplo de uso
if __name__ == "__main__":
    text = """
    La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas, 
    especialmente sistemas informáticos. Estos procesos incluyen el aprendizaje (la adquisición de información 
    y reglas para el uso de la información), el razonamiento (usando las reglas para llegar a conclusiones 
    aproximadas o definitivas) y la autocorrección. Aplicaciones particulares de la IA incluyen sistemas expertos, 
    reconocimiento de voz y visión artificial. La IA fue fundada como disciplina académica en 1956, y en los años 
    siguientes ha experimentado varias olas de optimismo, seguidas por desilusión y pérdida de financiación, 
    seguidas por nuevos enfoques, éxito y renovada financiación. La IA ha sido una rica fuente para muchas ideas 
    revolucionarias en informática. El aprendizaje profundo es parte del aprendizaje automático basado en un 
    conjunto de algoritmos que intentan modelar abstracciones de alto nivel en datos usando arquitecturas 
    computacionales que admiten transformaciones no lineales múltiples e iterativas de datos expresados en forma matricial o tensorial.
    """
    
    summary = generate_summary(text, 3)
    print("Texto original:")
    print(text)
    print("\nResumen generado:")
    print(summary)