import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler


st.set_page_config(
   page_title='BANKING MARKETING',
   page_icon='',
   layout='wide',
   menu_items={
       'Get Help': 'https://github.com/batuyildizz/banking_marketing',
       'Report a bug': 'https://www.example.com/help'
       
   }
)



st.image("C:/Users/batuy/OneDrive/Desktop/images/BANKA.jpg",width=350)

st.header("**Goal**: Predict if the client will subscribe a term deposit.")

st.title('Banking Marketing Project')






st.header("Data Dictionary")
st.markdown("- **subscribed**: has the client subscribed a term deposit? (0 = No, 1 = Yes)")
st.markdown("- **age**: Age of the person  ")
st.markdown("- **default**: has credit in default?")
st.markdown("- **balance**: average yearly balance")
st.markdown("- **housing**: has housing loan?")
st.markdown("- **loan**: has personal loan?")
st.markdown("- **day**: Square-feet of the house")
st.markdown("- **month**: last contact month of year")
st.markdown("- **duration**: last contact duration(in seconds)")
st.markdown("- **campaign**: number of contacts performed during this campaign and for this client")
st.markdown("- **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted)")
st.markdown("- **previous**: number of contacts performed before this campaign and for this client")


df = pd.read_csv("C:/Users/batuy/OneDrive/Desktop/bank-full.csv")


st.table(df.sample(11))
st.sidebar.markdown("**Choose** the features below to see the result!")


age = st.sidebar.number_input("Age",)
default = st.sidebar.selectbox("Default", [0, 1])  # 0 = No, 1 = Yes
balance = st.sidebar.number_input("Balance")
housing = st.sidebar.selectbox("Housing", [0, 1])  # 0 = No, 1 = Yes
loan = st.sidebar.selectbox("Loan", [0, 1])  # 0 = No, 1 = Yes
day = st.sidebar.slider("Day", min_value=1, max_value=31, value=1, help="Max value:31, min value:1")
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=1, help="Max value:12, min value: 1")
duration = st.sidebar.number_input("Duration")
campaign = st.sidebar.number_input("Campaign", min_value=1)
pdays = st.sidebar.number_input("Pdays", min_value=-1)
previous = st.sidebar.number_input("Previous", min_value=0)


rf_model = load("C:/Users/batuy/OneDrive/Desktop/rf_model.pkl")



input_data = np.array([[duration]])






prediction = rf_model.predict(input_data)
prediction_proba = rf_model.predict_proba(input_data)





st.header("Results")


if st.sidebar.button("Submit"):

    
    st.info("You can find the result below.")
   
    
    results_df = pd.DataFrame({
    'age': [age],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'day': [day],
    'month': [month],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'Duration': [duration],
    'Prediction': prediction,
    'Probability_No': prediction_proba[:, 0],
    'Probability_Yes': prediction_proba[:, 1]
})

    


    
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0", "failed").replace("1", "subscribed"))

   
    st.write(f"Prediction: {'subscribed' if prediction[0] == 1 else 'Failed'}")
    st.write(f"Probability: {round(prediction_proba[0][0] * 100, 2)}% No, {round(prediction_proba[0][1] * 100, 2)}% Yes")
    st.write("Results DataFrame:")
    st.dataframe(results_df)

    if prediction[0] == 1:
        st.image("C:/Users/batuy/OneDrive/Desktop/images/tik1.jpg",width=750)
    else:
        st.image("C:/Users/batuy/OneDrive/Desktop/images/carpi2.jpg",width=750)
else:
    st.markdown("Please click the *Submit Button*!")
