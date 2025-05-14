import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def textrank_summarize(text, num_sentences=5):
    # Tokenizar en oraciones
    sentences = sent_tokenize(text)
    
    # Crear matriz de similitud de oraciones usando TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calcular la similitud del coseno entre las oraciones
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Construir grafo
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Aplicar PageRank
    scores = nx.pagerank(nx_graph)
    
    # Ordenar oraciones por puntuación
    ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
    
    # Obtener las mejores n oraciones y ordenarlas por posición original
    top_n = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
    
    # Crear resumen
    summary = ' '.join([sentences[i] for _, i in top_n])
    
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
    
    summary = textrank_summarize(text, 3)
    print("Texto original:")
    print(text)
    print("\nResumen generado:")
    print(summary)