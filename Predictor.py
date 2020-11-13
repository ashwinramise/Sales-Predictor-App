'''
Author: Ashwin Ramachandran
Affiliation: North Carolina State University
-------------------------------------XXX------------------------------------
FUNCTIONS USED IN DASH APP TO PREDICT SALES
'''
import pandas as _pd
import numpy as _np
import joblib as _joblib
from sklearn.preprocessing import LabelEncoder as _le
import json as _json
from datetime import datetime as _datetime
import plotly.graph_objects as _go
from plotly.subplots import make_subplots as _make_subplots
import warnings as _warnings
import plotly.express as _px

_warnings.filterwarnings('ignore')

# ------------------------------------xxx--------------------------------------------------------------xxx---------------------

AllData = _pd.read_csv('.\DataModels\DataSets DO_NOT_DEL\ForQpred.csv')
AllData['Bill Date'] = _pd.to_datetime(AllData['Bill Date'],dayfirst=True)

# Comapre Predictions with Historical data and reset predictions accordingly
def _createstatedf(state, county, method):
    StateCounty = AllData[(AllData['State'] == state) & (AllData['County']== county)]
    StateCounty = StateCounty.sort_values('Bill Date')
    if method == 'PandemicRegressor':
        StateCounty = StateCounty[StateCounty['Year'] == 2020]
    else:
        pass
    return StateCounty


# Prediction of Existing data to have accuracy index
def _calculate_q_factor(state, county, method, brand=None,subbrand=None):
    df = _createstatedf(state, county, method)
    df = writecols(df, state, county, brand, subbrand)
    struct_data = df.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    le = _le()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    DataFrame = struct_data[
                ['State', 'County', 'Brand', 'Sub Brand', 'Bill Date',
       'Pandemic cases Cumulative', 'Months', 'Year']]
    model= _joblib.load(f'.\DataModels\State\Predict Sales\{state}\Pandemic_Sales.joblib')
    df['PandemicSales'] = model.predict(DataFrame)
    q = sum(df['PandemicSales'].to_list())/sum(df['Qty'].to_list())
    return q

# ------------------------------------xxx--------------------------------------------------------------xxx---------------------

# Load all the required dictionaries for encoding
def _definedicts(filepath):
    with open(filepath, 'r') as file:
        NewD = _json.load(file)
    return NewD


# All required dictionaries for loading
_County = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\Counties.json')
_State = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\States.json')
_Brand = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\Brands.json')
_SubBrand = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\SubBrands.json')
_StateQValsPan = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\StateQValsPan.json')
_StateQValsHist = _definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\StateQValsNhist.json')


# Cleanup Function for all Dates in a file
def _datetimeCleanup(df):
    Datalist=df['Bill Date'].to_list()
    Cleaned = []
    for date in Datalist:
        if '/' in date:
            try:
                p = _datetime.strptime(date,'%m/%d/%Y')
            except ValueError:
                try:
                    p = _datetime.strptime(date,'%d/%m/%Y')
                except ValueError:
                    p = _datetime.strptime(date,'%m/%d/%y')
        elif '-' in date:
            try:
                p = _datetime.strptime(date,'%m-%d-%Y')
            except ValueError:
                p = _datetime.strptime(date, '%d-%m-%Y')
        Cleaned.append(p)
    df['Bill Date'] = Cleaned
    return df


# Writing required columns to data frame
def writecols(DataFrame, state, county, brand=None,subbrand=None):
    df=DataFrame.copy()
    if brand is None and subbrand is None:
        df['Bill Date']= _pd.to_datetime(df['Bill Date'], dayfirst=True)
        df['State'] = _State.get(state)
        df['County'] = _County.get(county)
        df['Months'] = [date.month for date in df['Bill Date'].to_list()]
        df['Year'] = [date.year for date in df['Bill Date'].to_list()]
    else:
        df['Bill Date']= _pd.to_datetime(df['Bill Date'], dayfirst=True)
        df['State'] = _State.get(state)
        df['County'] = _County.get(county)
        df['Months'] = [date.month for date in df['Bill Date'].to_list()]
        df['Year'] = [date.year for date in df['Bill Date'].to_list()]
        df['Brand'] = _Brand.get(brand)
        df['Sub Brand']= _SubBrand.get(subbrand)
    return df


# Main Predictor Function
def predictor(df, state, county, method, brand=None,subbrand=None):
    '''Mehtod = Natural_Hist/PandemicSales'''
    Data = writecols(df, state, county, brand, subbrand)
    struct_data = Data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    le = _le()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    if method == 'Natural_Hist':
        Data = struct_data[['State', 'County', 'Bill Date', 'Months']]
        model= _joblib.load(f'.\DataModels\State\Predict Sales\{state}\History_Sales.joblib')
        Preds = model.predict(Data)
#         q = sum(Preds)/sum(df['Qty'].to_list())
        q = _StateQValsHist.get(state).get(county)
        df['Natural_Hist'] = [round(val/q) for val in Preds]
    elif method == 'PandemicSales':
        Data = struct_data[
                ['State', 'County', 'Brand', 'Sub Brand', 'Bill Date',
       'Pandemic cases Cumulative', 'Months', 'Year']]
        model= _joblib.load(f'.\DataModels\State\Predict Sales\{state}\Pandemic_Sales.joblib')
        k = model.predict(Data)
        q = _calculate_q_factor(state, county, method, brand,subbrand)
#         q = sum(k)/sum(df['Qty'].to_list())
        df['PandemicSales'] = [round(val/q) for val in k] # Reinforcement step using q index
    df['State'] = state
    df['County'] = county
    return df

      
# Plotting Sales Vs COVID Evolution
def sales_plot(df, over='PandemicSales',brand=None,subbrand=None):
    ''' df is the input data frame
    over = PandemicSales or Natural_Hist'''
    fig = _make_subplots(specs=[[{"secondary_y": True}]])
    if over == 'PandemicSales':
#         Create figure with secondary y-axis

        fig.add_trace(
            _go.Bar(x=df['Bill Date'], y=df['Pandemic cases Cumulative'], name="Pandemic Evolution", opacity=0.50),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Pandemic Evolution", secondary_y=True)      
    else:
        pass
# Add traces
#-------------------------FOR CHECKING ACCURACY OF PREDICTIONS------------------------------------------
#     fig.add_trace(
#         _go.Line(x=df['Bill Date'], y=df['PandemicSales'], name='PandemicSales'),
#         secondary_y=False,
#     )
#     fig.add_trace(
#         _go.Line(x=df['Bill Date'], y=df['Qty'], name="Actual Sales"),
#         secondary_y=False,
#     )
#-------------------------FOR CHECKING ACCURACY OF PREDICTIONS------------------------------------------
    fig.add_trace(
        _go.Line(x=df['Bill Date'], y=df[over], name=over),
        secondary_y=False,
     )

    # Add figure title
    if over == 'PandemicSales':
        fig.update_layout(
        title_text=f"{over} vs Sales in {df['State'].to_list()[0]},{df['County'].to_list()[0]}| {brand},{subbrand}", plot_bgcolor='rgba(0, 0, 0, 0)'
        )
    else:
        fig.update_layout(
        title_text=f"{over} vs Sales in {df['State'].to_list()[0]},{df['County'].to_list()[0]}", plot_bgcolor='rgba(0, 0, 0, 0)'
        )
# for accuracycheck title    
#     fig.update_layout(
#         title_text=f"PandemicSales vs Nat_History vs Actual Sales in {df['State'].to_list()[0]},{df['County'].to_list()[0]}", plot_bgcolor='rgba(0, 0, 0, 0)'
#     )

    # Set x-axis title
    fig.update_xaxes(title_text="Dates")

    # Set y-axes titles
    fig.update_yaxes(title_text="Sales Qty", secondary_y=False)

    fig.show()




def dateconvert(df):
    '''Converts any datetime format into a datetime object for easy inference and sorting
    input = DataFrame'''
    try:
        df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%m/%d/%y')
    except ValueError:
        try:
            df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%m/%d/%Y')
        except ValueError:
            try:
                df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%m-%d-%Y')
            except ValueError:
                try:
                    df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%d-%m-%Y')
                except ValueError:
                    try:
                        df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%m-%d-%y')
                    except ValueError:
                        try:
                            df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%d-%m-%y')
                        except ValueError:
                            try:
                                df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%d/%m/%Y')
                            except ValueError:
                                df['Bill Date'] = _pd.to_datetime(df['Bill Date'], format='%Y-%m-%d')
    return df