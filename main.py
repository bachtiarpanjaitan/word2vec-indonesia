import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

class Word2VecLoader:
    def load_word2vec(self, file_path):
        word_vectors = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                word_vectors[word] = vector
        return word_vectors


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def find_closest_words(target_word, word_vectors, top_n=5):
    if target_word not in word_vectors:
        print(f"Kata '{target_word}' tidak ditemukan di embedding! Menggunakan vektor nol sebagai fallback.")
        return [(target_word, 0.0)]

    target_vector = word_vectors[target_word]
    similarities = {
        word: cosine_similarity(target_vector, vector)
        for word, vector in word_vectors.items() if word != target_word
    }

    closest_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return closest_words


def build_embedding_matrix(word_index, word_vectors, vocab_length, embedding_dim):
    embedding_matrix = np.zeros((vocab_length, embedding_dim))

    for word, index in word_index.items():
        if index >= vocab_length:
            continue

        embedding_vector = word_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector / np.linalg.norm(embedding_vector)
        else:
            embedding_matrix[index] = np.random.uniform(-0.01, 0.01, embedding_dim)

    return embedding_matrix


if __name__ == "__main__":
    loader = Word2VecLoader()
    file_path = "word2vec.txt"  # Ganti dengan path yang sesuai

    word_vectors = loader.load_word2vec(file_path)

    # Tokenizer buat generate word_index
    teks = input("Masukkan kalimat Token: ")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([teks])
    word_index = tokenizer.word_index

    vocab_length = len(word_index) + 1
    embedding_dim = 50

    embedding_matrix = build_embedding_matrix(word_index, word_vectors, vocab_length, embedding_dim)

    input_word = input("Masukkan kata untuk cari yang mirip: ")
    closest_words = find_closest_words(input_word, word_vectors)

    if closest_words:
        print(f"Kata yang mirip dengan '{input_word}':")
        for word, score in closest_words:
            print(f"{word} (skor: {score:.4f})")

# Sekarang ada tokenizer otomatis yang bikin word_index dari teks input! 🚀 Yuk, cobain! ✨
