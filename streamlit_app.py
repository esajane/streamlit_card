import streamlit as st
import pandas as pd
import warnings
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# Load the scaler and the model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = load_model('model.keras')
rf_model = pickle.load(open('random_forest.pkl', 'rb'))
xgb_model = pickle.load(open('xgboost.pkl', 'rb'))

V_FEATURE_MEAN = [-0.20842597938263605, 0.07257380219465191, 0.7182821832180077, 0.1957470748220621, -0.2166917096436666, 0.09557486227318143, -0.11684695428019536, 0.032755436081655494, 0.2594590077281431, -0.08679933893728993, 0.45634495837618255, -0.5394356944092601, 0.29513922421173017, 0.26906988445813024, 0.08256303219514335, -0.0007689825786242458, 0.16560898362346058, -0.08743786166875443, -0.036865973821847656, 0.04481352065768667, -0.030876028108593082, -0.1135549060528877, -0.04157133779307225, 0.007468548218399575, 0.13595806261255644, 0.02181261633255956, 0.01083611902584234, 0.003833698553660782]

# Define the Streamlit interface
st.title('Credit Card Fraud Detection System')

st.header('Enter Basic Transaction Details:')
transaction_time = st.number_input('Transaction Time (seconds since first transaction)', )
transaction_amount = st.number_input('Transaction Amount',)

# Advanced options for V features
st.subheader('Advanced Options:')
with st.expander("PCA Components (V1-V28)"):
    st.markdown("""
    *The V1-V28 fields represent components obtained from a PCA transformation due to confidentiality. \
    If you are not familiar with these values, leave them as the default.*
    """)
    V_features = [st.number_input(f'V{i+1}', value=float(V_FEATURE_MEAN[i])) for i in range(28)]

# Button to predict
if st.button('Predict Fraud'):
    features =  features = [transaction_time, transaction_amount] + V_features
    features = np.array(features)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # Prediction
    prediction = rf_model.predict(features)
    probability = rf_model.predict_proba(features)[0][1]

    # Display results
    st.header('Results:')
    st.subheader(f'Fraud Probability: {probability:.2f}')
    if probability > 0.981194:
        st.error('This transaction is likely to be fraudulent.')
    else:
        st.success('This transaction is likely to be non-fraudulent.')
warnings.filterwarnings('ignore', category=UserWarning, message="X does not have valid feature names")
# `streamlit run streamlit_app.py` 
