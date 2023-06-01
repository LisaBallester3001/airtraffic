# Importation

import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from prophet import Prophet
from plotly.subplots import make_subplots
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from mlforecast import MLForecast
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
import matplotlib.pyplot as plt



# Insertion des données

HOME_AIRPORTS = ('LGW', 'LIS', 'LYS', 'SSA', 'NTE', 'PNH', 'POP')
PAIRED_AIRPORTS = ('FUE', 'AMS', 'ORY', 'BCN', 'PIS', 'OPO', 'NGB', 'JFK', 'GRU')
MODEL_NAMES = ['Prophet', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor']



# Ouverture du fichier

df = pd.read_parquet('src/data/traffic_10lines.parquet')



# Texte dans la barre à gauche : 

st.sidebar.markdown('<span style="font-size:18px;font-weight:bold;text-decoration:underline;">Data selection for air traffic forecasts :</span>', unsafe_allow_html=True) # Sous titre


with st.sidebar:
    # Sélection de l'aéroport de départ
    home_airport = st.selectbox('Home Airport', HOME_AIRPORTS, key='home_airport_select')
    st.write("<span style='font-size:14px;font-style:italic;'>By selecting the departure airport of your choice, you will be able to select only those paired airports with which you have data.</span>", unsafe_allow_html=True) # phrase de précision

    # Filtrage des aéroports d'arrivée disponibles en fonction de l'aéroport de départ sélectionné
    available_paired_airports = df.loc[df['home_airport'] == home_airport, 'paired_airport'].unique()
    paired_airport = st.selectbox('Paired Airport', available_paired_airports, key='paired_airport_select')

    # Choix de la date de début des prévisions
    forecast_date = st.date_input('Forecast Start Date', key='forecast_date_input')
    # Choix du nombre de jours de prévision : allant jusquu'à 365 jours
    nb_days = st.slider('Days of forecast', 7, 365, 1, key='nb_days_slider')

    # sous titre de la partie sélection du modèle
    st.markdown('<span style="font-size:18px;font-weight:bold;text-decoration:underline;font-style:italic;">Selecting the forecast model :</span>', unsafe_allow_html=True)

    # bouton de sélection du modèle
    model_name = st.selectbox('Model', MODEL_NAMES, key='model_select')

    # bouton pour afficher des prévisions sous forme de graphique
    Forecast_bouton = st.button('Forecast')
    st.markdown('<span style="font-size: 14px; line-height: 0.8; font-style: italic; margin-bottom: -10px;">The "Forecast" button will allow you to make a traffic prediction for the selected route</span>', unsafe_allow_html=True) # phrase de précision

# Texte sur la page :

st.title('Traffic Forecaster') # Titre

st.markdown('<span style="font-size:18px;font-weight:bold;text-decoration:underline;font-style:italic;">Data selected for air traffic forecasts :</span>', unsafe_allow_html=True) # Sous titre pour la sélection des données

# Affichage des différentes sélections faite dans la barre à gauche 

st.markdown(f'<div style="font-size:15px;"><span style="line-height:0.8;font-style:italic;color:navy;">Home Airport selected: {home_airport}</span></div>', unsafe_allow_html=True) 
st.markdown(f'<div style="font-size:15px;"><span style="line-height:0.8;font-style:italic;color:navy;">Paired Airport selected: {paired_airport}</span></div>', unsafe_allow_html=True)
st.markdown(f'<div style="font-size:15px;"><span style="line-height:0.8;font-style:italic;color:navy;">Days of forecast: {nb_days}</span></div>', unsafe_allow_html=True)
st.markdown(f'<div style="font-size:15px;"><span style="line-height:0.8;font-style:italic;color:navy;">Date selected: {forecast_date}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<span style="font-size:18px;font-weight:bold;text-decoration:underline;font-style:italic;">Press the "Dataframe" button to observe the data present in it</span>', unsafe_allow_html=True)

dataframe_bouton = st.button('Dataframe')
st.markdown('<span style="color: #808080; line-height: 0.8;font-size:10px; font-style: italic;">By pressing the button again the dataframe will be removed</span>', unsafe_allow_html=True)

if dataframe_bouton:
    if 'dataframe_shown' not in st.session_state:
        st.session_state.dataframe_shown = True
    else:
        st.session_state.dataframe_shown = not st.session_state.dataframe_shown

if 'dataframe_shown' in st.session_state and st.session_state.dataframe_shown:
    st.dataframe(data=df, width=600, height=300)

    
    
# Affichage du graphique en fonction des données sélectionnée dans la barre à gauche :

def draw_ts_multiple(df: pd.DataFrame, v1: str, v2: str=None, prediction: str=None, date: str='date',
                     secondary_y=True, covid_zone=False, display=True, home_airport=None, paired_airport=None):
    # Vérifier si les aéroports ont été spécifiés, sinon utiliser les aéroports du DataFrame
    if home_airport is None:
        home_airport = df['home_airport'].unique()[0]
    if paired_airport is None:
        paired_airport = df['paired_airport'].unique()[0]
    
    # Définition des variables à tracer
    variables = [(v1, 'V1')]
    # Construction du titre du graphique en fonction des aéroports
    title = f'Traffic from {home_airport} to {paired_airport}'
    if v2:
        variables.append((v2, 'V2'))
        title += f' - {v2}'
    
    # Définition de la mise en page du graphique
    layout = dict(
        title=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )
    
    # Création de la figure en utilisant make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(layout)
    
    # Ajout des traces pour chaque variable à tracer
    for v, name in variables:
        fig.add_trace(go.Scatter(x=df[date], y=df[v], name=name), secondary_y=False)
    
    # Configuration de l'axe y secondaire si une deuxième variable est spécifiée
    if v2:
        fig['layout']['yaxis2']['showgrid'] = False
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(margin=dict(t=125))
    
    # Ajout de la trace de prédiction si spécifiée
    if prediction:
        fig.add_trace(go.Scatter(x=df[date], y=df[prediction], name='^V1', line={'dash': 'dot'}), secondary_y=False)
    
    # Ajout de la zone COVID-19 si spécifiée
    if covid_zone:
        fig.add_vrect(
            x0=pd.Timestamp("2020-03-01"),
            x1=pd.Timestamp("2022-01-01"),
            fillcolor="Gray",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    
    # Affichage du graphique Plotly dans Streamlit
    if display:
        st.plotly_chart(fig)
    
    return fig





    
# Différents modèles de prévision :


# modèle de prévision Prophet

def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
    _df = (traffic_df
           .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
           .groupby(['home_airport', 'paired_airport', 'date'])
           .agg(pax_total=('pax', 'sum'))
           .reset_index()
           )
    return _df

def show_forecast_results(df: pd.DataFrame, home_airport: str, paired_airport: str):

    route_df = generate_route_df(df, home_airport, paired_airport)
    route_df = route_df.rename(columns={'date': 'ds', 'pax_total': 'y'})

    baseline_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    baseline_model.fit(route_df)

    forecast_end_date = forecast_date + pd.DateOffset(days=nb_days)
    future = pd.date_range(start=forecast_date, end=forecast_end_date, freq='D')

    forecast = baseline_model.predict(pd.DataFrame({'ds': future}))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=route_df['ds'], y=route_df['y'], name='Historical Traffic', mode='lines'), secondary_y=False)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Traffic', mode='lines'), secondary_y=False)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', name='Confidence Interval', mode='lines'), secondary_y=False)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', name='Confidence Interval', mode='lines'), secondary_y=False)

    fig.update_layout(title_text=f'Traffic from {home_airport} to {paired_airport}', xaxis_rangeslider_visible=True)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Traffic', secondary_y=False)

    fig.update_layout(xaxis_rangeslider=dict(visible=True))

    fig.add_vrect(
        x0=pd.Timestamp("2020-03-01"),
        x1=pd.Timestamp("2022-01-01"),
        fillcolor="Gray",
        opacity=0.5,
        layer="below",
        line_width=0,
    )

    st.plotly_chart(fig)
    
    
    
# modèle de prévision Nixtla

k = nb_days

tested_models = [
    lgb.LGBMRegressor(),
    xgb.XGBRegressor(),
    RandomForestRegressor(random_state=0),
]

@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)


fcst = MLForecast(
    models=tested_models,
    freq='D',
    lags=[7, 14, 21, 28],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    differences=[1],
)

nixtla_model = fcst.fit(generate_route_df(df, home_airport, paired_airport).drop(columns=['paired_airport']),
                        id_col='home_airport', time_col='date', target_col='pax_total')

predict_df = nixtla_model.predict(k)






# Affichage des données à la suite de la sélection des boutons : 


if home_airport and paired_airport and forecast_date and nb_days and not Forecast_bouton:
    st.markdown('<span style="font-size: 18px; font-weight: bold; text-decoration: underline;">Date-based air traffic display : </span> <span style="font-size: 18px; font-weight: bold; text-decoration: underline;"></span>', unsafe_allow_html=True)
    # Affichage du graphique en utilisant la fonction draw_ts_multiple avec les aéroports sélectionnés
    draw_ts_multiple(
        (df
         .query('home_airport == @home_airport and paired_airport == @paired_airport')  # Utilisation des variables sélectionnées
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
        ),
        'pax_total',
        covid_zone=True,
        home_airport=home_airport,  # Passage des aéroports sélectionnés
        paired_airport=paired_airport,
    )

if home_airport and paired_airport and forecast_date and nb_days and Forecast_bouton:
    st.markdown('<span style="font-size: 18px; font-weight: bold; text-decoration: underline;">Display of air traffic according to date as well as air forecast depending on selected data:</span> <span style="font-size: 18px; font-weight: bold; text-decoration: underline;"></span>', unsafe_allow_html=True)
    st.markdown('<span style="font-size: 16px;text-decoration: underline;">Model selected: </span><span style="font-size: 16px; ">{}</span>'.format(model_name), unsafe_allow_html=True)



    if model_name == 'Prophet':
        show_forecast_results(df, home_airport, paired_airport)

    if model_name == 'LGBMRegressor' or model_name == 'XGBRegressor' or model_name == 'RandomForestRegressor':
        fig, ax = plt.subplots(figsize=(15, 7))
        (pd.concat([generate_route_df(df, home_airport, paired_airport).drop(columns=['paired_airport']),
                nixtla_model.predict(k)])
         .set_index('date')
         ).plot(ax=ax)

        ax.set_title(f'Traffic from {home_airport} to {paired_airport}')

        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-01-01"), facecolor='gray', alpha=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Traffic')

        st.pyplot(fig)


# le code de l'affichage des graphiques n'est vraiment pas optimisé mais je me suis rendu compte de mon erreur trop tard pour avoir le temps de tout modifier je suis désolée...



#Display of air traffic according to date as well as air forecast depending on selected data




