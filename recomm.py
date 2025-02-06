import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
users = pd.read_csv('users.csv')
posts = pd.read_csv('posts.csv')
interactions = pd.read_csv('interactions.csv')

# Preprocess data
tfidf = TfidfVectorizer()
post_tags_tfidf = tfidf.fit_transform(posts['Tags'])

interaction_mapping = {
    "View": 1,
    "Like": 2,
    "Comment": 3,
    "Share": 4
}
interactions["Interaction_Type"] = interactions["Interaction_Type"].map(interaction_mapping)
interactions = interactions.dropna(subset=["Interaction_Type"])
interactions["Interaction_Type"] = interactions["Interaction_Type"].astype(int)

interaction_matrix = interactions.pivot_table(
    index='User_ID',
    columns='Post_ID',
    values='Interaction_Type',
    fill_value=0,
    aggfunc='sum'
)

svd = TruncatedSVD(n_components=50)
user_embeddings = svd.fit_transform(interaction_matrix)
post_embeddings = svd.components_.T

# Content-based filtering function
def content_based_recommendations(user_id):
    user_interests = users[users['User_ID'] == user_id]['Interests'].values[0]
    user_interests_tfidf = tfidf.transform([user_interests])
    similarities = cosine_similarity(user_interests_tfidf, post_tags_tfidf).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    return posts.iloc[top_indices]

# Collaborative filtering function
def collaborative_score(user_id, post_id):
    user_index = interaction_matrix.index.get_loc(user_id)
    post_index = interaction_matrix.columns.get_loc(post_id)
    return user_embeddings[user_index].dot(post_embeddings[post_index])

# Hybrid recommendation function
def hybrid_recommendations(user_id):
    try:
        content_recs = content_based_recommendations(user_id)
        
        # Calculate cosine similarities for the recommended posts
        similarities = cosine_similarity(tfidf.transform([users[users['User_ID'] == user_id]['Interests'].values[0]]), 
                                         post_tags_tfidf[content_recs.index]).flatten()
        
        # Add cosine similarities to DataFrame
        content_recs['Cosine_Similarity'] = similarities
        
        # Calculate collaborative scores for recommended posts
        collaborative_scores = [
            collaborative_score(user_id, post_id) for post_id in content_recs['Post_ID']
        ]
        
        # Add collaborative scores to DataFrame
        content_recs['Collaborative_Score'] = collaborative_scores
        
        # Compute Hybrid Score
        content_recs['Hybrid_Score'] = (0.6 * content_recs['Cosine_Similarity'] + 
                                        0.4 * content_recs['Collaborative_Score'])
        
        return content_recs.sort_values(by='Hybrid_Score', ascending=False).head(3)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

# Evaluation function (if labeled data is available)
def evaluate_recommendations(recommendations, ground_truth):
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return {"Precision": 0, "Recall": 0, "F1-Score": 0}
    
    # Convert everything to strings
    rec_ids = set(str(post_id) for post_id in recommendations['Post_ID'])
    true_ids = set(str(post_id) for post_id in ground_truth['Post_ID'])
    
    # Binary classification: 1 if post is in recommendations, 0 otherwise
    y_true = [1 if str(p) in true_ids else 0 for p in rec_ids]
    y_pred = [1 if str(p) in rec_ids else 0 for p in rec_ids]
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {"Precision": precision, "Recall": recall, "F1-Score": f1}
    
    

# Streamlit UI
st.title("Social Media Recommendation System")

selected_user = st.selectbox("Select a User", users['User_ID'].unique())

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(selected_user)
    
    if not recommendations.empty:
        st.subheader("Recommended Posts")
        for _, row in recommendations.iterrows():
            st.write(f"**Title**: {row['Title']}")
            st.write(f"**Category**: {row['Category']}")
            st.write(f"**Description**: {row.get('Description', 'No description available')}")
            st.write(f"**Tags**: {row['Tags']}")
            if 'Image_URL' in row and pd.notna(row['Image_URL']):
                st.image(row['Image_URL'], caption=row['Title'], use_container_width=True)
            st.write("---")
            
            # Reason for recommendation
            st.write(f"**Reason**: Recommended based on similarity to your interests ('{users[users['User_ID'] == selected_user]['Interests'].values[0]}')")
        
        # Evaluation metrics
        st.subheader("Evaluation Metrics")
        ground_truth = posts[posts['Post_ID'].isin(interactions[interactions['User_ID'] == selected_user]['Post_ID'])]
        
        # Convert Post_ID to strings in recommendations and ground_truth
        recommendations['Post_ID'] = recommendations['Post_ID'].astype(str)
        ground_truth['Post_ID'] = ground_truth['Post_ID'].astype(str)
        
        eval_metrics = evaluate_recommendations(recommendations, ground_truth)
        st.write(f"Precision: {eval_metrics['Precision']:.2f}")
        st.write(f"Recall: {eval_metrics['Recall']:.2f}")
        st.write(f"F1-Score: {eval_metrics['F1-Score']:.2f}")
        
       # Bar chart visualization
        st.subheader("User Interaction Summary")
        user_interactions = interactions[interactions['User_ID'] == selected_user]
        interaction_mapping_reverse = {1: "View", 2: "Like", 3: "Comment", 4: "Share"}
        user_interactions['Interaction_Name'] = user_interactions['Interaction_Type'].map(interaction_mapping_reverse)
        interaction_counts = user_interactions['Interaction_Name'].value_counts()

        fig, ax = plt.subplots()
        interaction_counts.plot(kind='bar', ax=ax)
        ax.set_title("User Interaction Types")
        ax.set_ylabel("Count")
        ax.set_xlabel("Interaction Type")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No recommendations found for this user.")