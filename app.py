import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Classification of space object")

#feature :'redshift', 	'u', 	'i' ,	'g', 	'r','z'
# Input number 1
redshift = st.number_input("Enter redshift : 0-1")

# Input slider 
u = st.slider("Enter U" , 0, 100)
i = st.slider("Enter I" , 0, 100)
g = st.slider("Enter G" , 0, 100)
r = st.slider("Enter R" , 0, 100)
z = st.slider("Enter Z" , 0, 100)

# If button is pressed
if st.button("Predict"):
    
    # Unpickle classifier
    model = joblib.load("model.pkl")
    
    # Store inputs into dataframe
    x = pd.DataFrame([[redshift, u, i, g, r, z]], 
                     columns = ["redshift", "u", "i", "g", "r", "z"])
    
    
    # Get prediction
    prediction = model.predict(x)[0]
    
    # Output prediction
    st.code(f"This object is a {prediction}")
