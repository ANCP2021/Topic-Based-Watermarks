import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class EmbeddingMapper:
    def __init__(
            self, 
            tokenizer, 
            sentence_model
    ):
        
        self.tokenizer = tokenizer
        self.sentence_model = sentence_model
    
    def get_model_vocab_embeddings(self):
        vocab = self.tokenizer.get_vocab()  
        tokens = list(vocab.keys())
        vocab_embeddings = self.sentence_model.encode(tokens, convert_to_tensor=True)  
        return tokens, vocab_embeddings
    
    def get_defined_topic_list_embeddings(self, topic_list):
        topic_embeddings = self.sentence_model.encode(topic_list, convert_to_tensor=True)  
        return topic_embeddings
    
    """
    Maps vocabulary tokens to topics based on cosine similarity and distributes tokens evenly among lists.
    """
    def map_tokens_to_topics(self, total_tokens, vocab_embeddings, defined_topics, topic_embeddings, threshold=0.7):
        topic_embeddings_np = topic_embeddings.detach().cpu().numpy()
        vocab_embeddings_np = vocab_embeddings.detach().cpu().numpy()

        topic_token_mapping = {t: [] for t in defined_topics}
        topic_above_threshold = {t: [] for t in defined_topics}
        topic_below_threshold = {t: [] for t in defined_topics}
        below_threshold_tokens= []
        string_to_id = self.tokenizer.get_vocab()

        for idx, token_str in enumerate(total_tokens):
            token_embedding = vocab_embeddings_np[idx].reshape(1, -1)

            # Cosine similarity with each topic embedding
            similarities = cosine_similarity(token_embedding, topic_embeddings_np).flatten()
            
            # Find the topic with the maximum similarity
            max_similarity = similarities.max()
            topic_index = similarities.argmax()
            if max_similarity >= threshold: # Assign to the topic if above threshold
                assigned_topic = defined_topics[topic_index]
                token_id = string_to_id[token_str]
                topic_token_mapping[assigned_topic].append(token_id)
                topic_above_threshold[assigned_topic].append(token_id)

            else: # Tokens below the threshold
                below_threshold_tokens.append(token_str)
                
        print("Above threshold token counts per topic:")
        for topic in defined_topics:
            print(f"{topic}: {len(topic_above_threshold[topic])}")

        print("Total tokens below threshold (before distribution):", len(below_threshold_tokens))

        # Distribute tokens below threshold evenly
        random.shuffle(below_threshold_tokens)
        num_topics = len(defined_topics)
        for i, token_str in enumerate(below_threshold_tokens):
            topic = defined_topics[i % num_topics]
            token_id = string_to_id[token_str]
            topic_token_mapping[topic].append(token_id)
            topic_below_threshold[topic].append(token_id)
            
            
        print("\nBelow threshold token counts per topic (after distribution):")
        for topic in defined_topics:
            print(f"{topic}: {len(topic_below_threshold[topic])}")

        # Print the final total token counts per topic
        print("\nFinal total token counts per topic:")
        for topic in defined_topics:
            print(f"{topic}: {len(topic_token_mapping[topic])}")    

        # Ensure even distribution of above-threshold tokens for positional bias
        for topic in defined_topics:
            random.shuffle(topic_token_mapping[topic])

        return topic_token_mapping
    
    """
    Average detected topics and compare with predefined topics for best match.
    """
    def detected_topics_to_embeddings(self, topic_list, detected_topics):
        topic_embeddings = self.get_defined_topic_list_embeddings(topic_list)
        detected_embeddings = self.get_defined_topic_list_embeddings(detected_topics)

        average_detected_embedding = detected_embeddings.mean(dim=0, keepdim=True)
        average_detected = average_detected_embedding.detach().cpu().numpy()
        topic_embedding = topic_embeddings.detach().cpu().numpy()

        similarities = cosine_similarity(average_detected, topic_embedding).flatten()
        best_idx = similarities.argmax()
        best_topic = topic_list[best_idx]

        return best_topic, similarities
    
    """
    K-means detected topics and compare with predefined topics for best match.
    """
    def kmeans_detected_topics_to_embeddings(self, topic_list, detected_topics):        
        topic_embeddings = self.get_defined_topic_list_embeddings(topic_list) 
        detected_embeddings = self.get_defined_topic_list_embeddings(detected_topics)

        topic_embeddings_np = topic_embeddings.detach().cpu().numpy()
        detected_embeddings_np = detected_embeddings.detach().cpu().numpy()

        # K-means clustering on detected topics
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(detected_embeddings_np)
        centroids = kmeans.cluster_centers_

        best_topic = None
        best_similarity_score = -float('inf')
        best_similarities = None
        # For each centriod find the closest predefined topic and keep track of best one
        for centroid in centroids:
            centroid_reshaped = centroid.reshape(1, -1)
            similarities = cosine_similarity(centroid_reshaped, topic_embeddings_np).flatten()
            max_similarity = similarities.max()
            if max_similarity > best_similarity_score:
                best_similarity_score = max_similarity
                best_topic = topic_list[similarities.argmax()]
                best_similarities = similarities

        return best_topic, best_similarities
    