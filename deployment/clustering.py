import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load all files
with open('list_num_cols.txt', 'r') as file_1:
  list_num_cols = json.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  scaler = pickle.load(file_2)
  
with open('model_pca.pkl', 'rb') as file_3:
  pca = pickle.load(file_3)

with open('model_kmeans.pkl', 'rb') as file_4:
  model_kmeans = pickle.load(file_4)
  
def run():
    st.write('##### Form Clustering Customer Segmentation')
    # Making Form
    with st.form(key='Form Clustering Customer Segmentation'):
        Customer_ID                = st.number_input('Customer Id', min_value=1, max_value=10000, value=1)
        Age                        = st.number_input('Age', min_value=20, max_value=60, value=20)
        Edu                        = st.number_input('Edu', min_value=1, max_value=5, value=1)
        Years_Employed             = st.number_input('Years Employed', min_value=1, max_value=55, value=1)
        Income                     = st.number_input('Income', min_value=0.0, max_value=9999.0, value=0.0)
        Card_Debt                  = st.number_input('Card Debt', min_value=0.0, max_value=9999.0, value=0.0)
        Other_Debt                 = st.number_input('Other Debt', min_value=0.0, max_value=9999.0, value=0.0)
        Defaulted                  = st.radio("Defaulted",["0", "1"])
        DebtIncomeRatio            = st.number_input('DebtIncomeRatio', min_value=0.0, max_value=100.0, value=0.0)
        st.markdown('---')
        
        submited = st.form_submit_button('Process Now')
        
        data_inf = {
            'Customer_ID'      : Customer_ID,
            'Age'              : Age,
            'Edu'              : Edu,
            'Years Employed'   : Years_Employed,
            'Income'           : Income,
            'Card Debt'        : Card_Debt,
            'Other Debt'       : Other_Debt,
            'Defaulted'        : Defaulted,
            'DebtIncomeRatio'  : DebtIncomeRatio
        }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submited:
        # Split between numerical columns and categorical columns
        data_inf_num = data_inf[list_num_cols]
        # Feature scaling
        data_inf_num_scaled  = scaler.transform(data_inf_num)
        data_inf_final = data_inf_num_scaled
        # PCA
        data_inf_num_pca  = pca.transform(data_inf_final)
        data_inf_final_pca = data_inf_num_pca
        # Predict using K-Means Clustering Model
        prediction_cluster = model_kmeans.predict(data_inf_final_pca)
        st.write('# Nasabah ini cocok masuk ke dalam klaster  : ', str(prediction_cluster[0]))
        if prediction_cluster == 0:
          st.write('Produk yang cocok untuk nasabah klaster 0 adalah sebagai berikut:')
          st.write('1. Tabungan Pendidikan')          
          st.write('2. Tabungan Masa Depan')          
        else:
          st.write('''Produk yang cocok untuk nasabah klaster 1 adalah sebagai berikut:
                    1. Tabungan Pendidikan
                    2. Tabungan Masa Depan''')
        
if __name__ == '__main__':
    run()