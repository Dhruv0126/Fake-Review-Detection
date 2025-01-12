# Fake Review Detection
Overview
Fake reviews can mislead customers and harm businesses. This project aims to detect fake reviews using machine learning algorithms. It includes a responsive website where users can analyze reviews or upload datasets to train the model further. The project uses Flask for the backend, integrates a trained AI model, and provides a dynamic, user-friendly interface.

Features
1. Signup/Login Page
Collects user details (First Name, Last Name, Email).
Redirects authenticated users to the Fake Review Detection Page.
2. Main Detection Page
Review Analysis: Enter a review to check if it's real or fake. The model provides the percentage of "OR" (Original Review) and "CG" (Counterfeit Generated).
CSV Upload: Upload a dataset to retrain the AI model.
Logout: Ends the session and redirects users to the login page.
3. AI Integration
Uses a trained model (e.g., SVC with TF-IDF pipeline) for detecting fake reviews.
Supports LIME and SHAP for explainable AI.
Technologies Used
Frontend:
HTML5, CSS3
JavaScript
Responsive Design with tech-themed animations
Backend:
Flask
Python (for AI/ML integration)
Machine Learning:
SVM (Support Vector Machine) trained on review datasets
TF-IDF for text vectorization
Explainable AI:
LIME (Local Interpretable Model-agnostic Explanations)
SHAP (SHapley Additive exPlanations)
