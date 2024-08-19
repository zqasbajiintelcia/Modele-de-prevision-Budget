import streamlit as st
import re
import hashlib
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st
from itertools import product
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from joblib import Parallel, delayed
import altair as alt
import time
import requests
from streamlit_lottie import st_lottie

# Suppress all warnings
warnings.filterwarnings("ignore")

#Page config
st.set_page_config(layout="wide")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
           .css-vl8c1e {backdrop-filter: none;}
            .css-12ttj6m {
                            border: none;
                            padding: 0;
                        }
            </style>

            """
st.markdown(hide_st_style, unsafe_allow_html=True)


#User selection initialisation
user_selection=None

@st.cache_data
# Load and prepare data
def load_and_prepare_data(filename):
    data = pd.read_csv(filename, parse_dates=['Id_date_agr'])
    data['Id_date_agr'] = pd.to_datetime(data['Id_date_agr'], format='%Y%m')
    data = data.sort_values(by='Id_date_agr', ascending=False)
    data.set_index('Id_date_agr', inplace=True)
    data.index = data.index.to_period('M').to_timestamp('M')
    return data

data = load_and_prepare_data('../datasets/final_pnl_gl_dataset-12-08-24_aggregated_per_month.csv')


# Function to filter data based on user selection
def filter_data(data, user_selection):
    if user_selection is not None and list(user_selection.values())[1:] != [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]:
        filtered_data = pd.DataFrame()


        for key, value in list(user_selection.items())[1:]:

            if value is not None and value != []:
                filtered_data = pd.concat([filtered_data, data[data[key].isin(value) ].groupby('Id_date_agr').agg({
            'Montant(€)': 'sum',
            'Montant 7€': 'sum'
            })])
                
        filtered_data=filtered_data.groupby('Id_date_agr').agg({
            'Montant(€)': 'sum',
            'Montant 7€': 'sum'
            })
    else:
        filtered_data = data.groupby('Id_date_agr').agg({
            'Montant(€)': 'sum',
            'Montant 7€': 'sum'
            })
    print(filtered_data.shape)    
    return filtered_data




# Function to get unique values for dropdowns based on filtered data
def get_unique_values(data, key):
    return [None] + sorted(data[key].unique().tolist())

# Function to train SARIMA model with parameters
def train_sarima_model(data, order, seasonal_order):
    model = SARIMAX(data['Montant(€)'], 
                    order=order, 
                    seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results

# Function to evaluate SARIMA model
def evaluate_model(model, data):
    forecast = model.get_forecast(steps=len(data))
    forecast_mean = forecast.predicted_mean
    mae = mean_absolute_error(data['Montant(€)'], forecast_mean)
    r2 = r2_score(data['Montant(€)'], forecast_mean)
    return mae, r2

# Function to perform grid search
def grid_search(data, p_values, d_values, q_values, P_values, D_values, Q_values, S_values, n_jobs=-1):
    best_aic = float('inf')
    best_mae = float('inf')
    best_params = None
    best_model = None
    results_cache = {}
    
    def evaluate_combination(p, d, q, P, D, Q, S):
        if (p,d,q)==(0,0,0) or (P, D, Q)==(0,0,0): return None, None, None, None, None, None
        seasonal_order = (P, D, Q, S)
        order = (p, d, q)
        if (order, seasonal_order) in results_cache:
            return results_cache[(order, seasonal_order)]
        try:
            model = train_sarima_model(data, order, seasonal_order)
            mae, r2 = evaluate_model(model, data)
            aic = model.aic
            results_cache[(order, seasonal_order)] = (model, mae, r2, aic, order, seasonal_order)
            return model, mae, r2, aic, order, seasonal_order
        except Exception as e:
            print(f"Model fitting failed for order={order} and seasonal_order={seasonal_order}: {e}")
            return None, None, None, None, None, None
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(p, d, q, P, D, Q, S)
        for p, d, q, P, D, Q, S in product(p_values, d_values, q_values, P_values, D_values, Q_values, S_values)
    )

    for model, mae, r2, aic, order, seasonal_order in results:
        if model is not None and aic < best_aic and mae < best_mae:
            best_aic = aic
            best_mae = mae
            best_model = model
            best_params = {'order': order, 'seasonal_order': seasonal_order}
    
    return best_model, best_params

# Function to save model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to load model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()




# Function to forecast with trained model
def forecast_with_model(model, steps=12):
    forecast = model.get_forecast(steps=steps)
    
    forecast_mean = forecast.predicted_mean
    forecast_index = forecast_mean.index
    forecast_band_inf  = forecast.conf_int().iloc[:, 0]
    forecast_band_sup  = forecast.conf_int().iloc[:, 1]

    return pd.Series(forecast_mean, index=forecast_index) , pd.Series(forecast_band_inf, index=forecast_index), pd.Series(forecast_band_sup, index=forecast_index) 

# Function to plot forecast
def plot_forecast(data, forecast_elems, title):

    chart = st.empty()

    forecast_df = pd.DataFrame({'Date': forecast_elems[0].index, 'Montant(€)': forecast_elems[0], 'lower' : forecast_elems[1], 'upper' : forecast_elems[2]})
    forecast_df['Type'] = 'Forecast'
    
    historical_df = pd.DataFrame({'Date': data.index, 'Montant(€)': data['Montant(€)']})
    historical_df['Type'] = 'Historical'

    history = alt.Chart(historical_df, title='Forecast').mark_line(color='blue').encode(
        alt.X('Date:T', scale=alt.Scale(zero=False)),
        alt.Y('Montant(€):Q', scale=alt.Scale(zero=False))).properties(
        height=360, width=720).interactive()
    chart.altair_chart(history, use_container_width=True)

    for i in range(1,len(forecast_df)+1):
        new_points = alt.Chart(forecast_df[0:i], title='Forecast').mark_line(
        point=alt.OverlayMarkDef(color="green") ,
        color='green').encode(
                              alt.X('Date:T', scale=alt.Scale(zero=False)),
                              alt.Y('Montant(€)', scale=alt.Scale(zero=False))).properties(
                              height=360, width=720).interactive()
        
        # Confidence Interval Band
        band = alt.Chart(forecast_df[0:i]).mark_area(color='red',
                        opacity=0.1
                    ).encode(
                        x='Date:T',
                        y='lower',
                        y2='upper'
                    )
        chart.altair_chart(history + new_points + band, use_container_width=True)
        time.sleep(0.25)









def main_prediction(data,user_selection):
    # Filter data based on user selections
    filtered_data = filter_data(data, user_selection)


    if user_selection is not None:
    # Define model filename based on user selection
        model_filename = f"model_{'_'.join([f'{k}_{v}' for k, v in list(user_selection.items())[1:] if v])}.pkl"
        model_filename = re.sub(r'[\',\[\/\\\:\*\?\"<>\|\]]', '_', model_filename)
        model_filename = model_filename.strip()
        model_filename = hashlib.sha256(model_filename.encode())

        model_filename = model_filename.hexdigest()+".pkl"
    else: 
        model_filename = "model_.pkl"


    try:
        # Try to load existing model
        best_model = load_model(model_filename)
        print(f"Loaded model from {model_filename}")
    except FileNotFoundError:
        # Hyperparameter grid
        p_values = [0, 1, 2]
        d_values = [0, 1, 2]
        q_values = [0, 1, 2]
        P_values = [0, 1, 2]
        D_values = [0, 1, 2]
        Q_values = [0, 1, 2]
        S_values = [12]

        # Perform grid search
        print("Performing grid search...")
        best_model, best_params = grid_search(filtered_data, p_values, d_values, q_values, P_values, D_values, Q_values, S_values)

        if best_model:
            print(f"Best model parameters: {best_params}")

            # Save the best model
            save_model(best_model, model_filename)
            print(f"Model saved as {model_filename}")

            
        else:
            print("No suitable model found.")


    # Forecast
    forecast_elms = forecast_with_model(best_model, user_selection['steps'])

    # Plot results
    if user_selection is not None:
        plot_forecast(filtered_data, forecast_elms, ' / '.join([f'{k}: {v}' for k, v in list(user_selection.items())[1:] if v and user_selection]))
    else:
        plot_forecast(filtered_data, forecast_elms, ' / all data')


# selection_columns = ['Code journal', 'No séquence', 'Id_Rub', 'ID_Compte', 'ID_Cc',
#        'ID_Site', 'ID_Soc', 'Rubrique_Altice', 'Société_IC',
#        'Filter_Type', 'id_Rub_Rep', 'Rubrique_Altice_Réallo',
#        'ID_Rubrique_Altice', 'Id_Client', 'Id_Projet', 'Id_Zone',
#        ]

selection_columns = ['Code journal', 'Id_Rub', 'ID_Compte', 'ID_Cc',
       'ID_Site', 'ID_Soc', 'Rubrique_Altice', 'Société_IC',
       'Filter_Type', 'id_Rub_Rep', 'Rubrique_Altice_Réallo',
       'ID_Rubrique_Altice', 'Id_Client', 'Id_Projet', 'Id_Zone',
       ]



# Define options for the dropdown lists

options = {f"{selection_columns[i]}": ["All"] + data[selection_columns[i]].unique().tolist() for i in range(len(selection_columns))}


# Streamlit application layout
st.title("Prévoir le budget future")


with st.form(key='my_form'):
    # Slider input
    slider_value = st.slider("Choisir le mois future à prévoir", min_value=1, max_value=12, step=1)

    # Expanders for dropdowns
    dropdown_values = {}

    with st.expander("Appliquer des filtres spécifiques"):

    
        cols = st.columns(4)
        
        for i, (col_name, values) in enumerate(options.items()):
            with cols[i % 4]:
                selected_values = st.multiselect(col_name, options=values)
                
                # If "All" is selected, select all other options
                if "All" in selected_values:
                    selected_values = values[1:]  # Exclude "All" from the list
                
                dropdown_values[col_name] = selected_values
        submit_button = st.form_submit_button(label='Valider les filtres')

    if submit_button:
        st.write("Selections confirmed!")

        # Store user selections in a dictionary
        user_selection = {
            'steps': slider_value,
            **dropdown_values
        }

        # Display the selected values
        st.subheader("Selected Values")
        st.write(user_selection)
        # Lottie animation URL
        lottie_url = "https://assets7.lottiefiles.com/packages/lf20_lchqwoif.json"
        lottie_animation = load_lottieurl(lottie_url)


        
        lottie_url = "https://lottie.host/0369e3df-c931-43ca-8372-d1ea94ed4c57/YrO04ImlAc.json"
        lottie_animation = load_lottieurl(lottie_url)

    

            
        with st.empty():
            st_lottie(lottie_animation, height=200, width=200)
            main_prediction(data, user_selection)
        

