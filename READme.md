# 🇮🇳 GUVI Multilingual Chatbot

A multilingual assistant for GUVI course information and career guidance, supporting 9 Indian languages with a Knowledge Base retrieval system and AI fallback.

## 🚀 Features
- Auto language detection and translation (9 Indian languages)
- Knowledge Base retrieval via FAISS
- AI fallback with fine-tuned FLAN model
- Career guidance & course recommendation logic
- Debug panel for live monitoring
- Clean Indian-themed UI

## Dataset and Model Fine-Tuning Summary

The GUVI Multilingual Chatbot leverages a domain-specific knowledge base (KB) consisting of **846 entries**, totaling approximately **50,800 tokens** with an average of 60 tokens per entry. This KB serves as the primary source for answering user queries related to GUVI courses and offerings, providing reliable and contextually relevant information through semantic search.

In addition, a fine-tuning dataset was created to adapt the FLAN-T5 base model for the GUVI domain. This dataset contains **1,169 brief examples** totaling roughly **2,300 tokens** (approximately 2 tokens per example), classifying the fine-tuning effort as **small-scale**. Due to its limited size, the fine-tuning dataset currently provides minor adjustments to the model and offers limited improvement in handling out-of-domain queries.

### Project Classification and Future Work

- **Scale:** Small-scale fine-tuning project  
- **Capabilities:** Strong KB-driven responses; fallback model lightly adapted to GUVI content  
- **Future plans:**  
  - Expand the fine-tuning dataset by collecting additional GUVI course materials, FAQs, learner queries, and publicly available educational resources.  
  - Consider data augmentation methods such as synthetic Q&A generation to enrich the diversity and coverage.  
  - Incrementally retrain and validate the model using medium-sized datasets (100K+ tokens) to enhance the chatbot’s intelligence and comprehension across a broader range of queries.

This approach ensures the chatbot remains effective for its core domain while paving the way for improved AI fallback responses and seamless multilingual support.

## 📦 Installation
