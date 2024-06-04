import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import json
with open('my_dict.json', 'r', encoding='utf-8') as f:
    my_dict = json.load(f)

label_encoder = LabelEncoder()


# Load your dataset or define input columns (features for prediction)
# data = ...

excel_file = 'data_filtered.xlsx'
xls = pd.ExcelFile(excel_file, engine='openpyxl')  # Specify engine if needed
data = pd.read_excel(xls, sheet_name='Sheet1')


def load_model():
    model = XGBRegressor()
    model.load_model('xgboost_model_streamlit.json')
    return model

model = load_model()

# Function to make predictions
def predict(list_):
    y_pred = model.predict([list_])
    return y_pred

# Streamlit app
def main():
    st.title('Car Price Prediction')
    st.write('Enter values according to given statements:')
    
    
    a_g = st.number_input('At Gucu', min_value=int(data['a.g.'].min()), max_value=int(data['a.g.'].max()), step=int(data['a.g.'].min()), value=int(data['a.g.'].min()))
    
    category_selected_model = st.selectbox('Model', data['Model'].unique())
    for i in range(len(my_dict[4][1])):
        if my_dict[4][1][i] == category_selected_model:
            category_encoded_model = my_dict[5][1][i]
   
    
    buraxilis_ili = st.number_input('Buraxılış ili', min_value=int(data['Buraxılış ili'].min()), max_value=int(data['Buraxılış ili'].max()), step=int(data['Buraxılış ili'].min()), value=int(data['Buraxılış ili'].min()))


    category_selected_marka = st.selectbox('Marka', data['Marka'].unique())
    for i in range(len(my_dict[2][1])):
        if my_dict[2][1][i] == category_selected_marka:
            category_encoded_marka = my_dict[3][1][i]
            

    litr = st.number_input('Litr', min_value=float(data['Litr'].min()), max_value=float(data['Litr'].max()), step=float(data['Litr'].min()), value=float(data['Litr'].min()))



    list_ = [a_g, category_encoded_model, buraxilis_ili, category_encoded_marka, litr]
    if st.button('Predict'):
        predicted_value = predict(list_)
        st.write(f'Predicted price for given values is {predicted_value} AZN')

if __name__ == '__main__':
    main()