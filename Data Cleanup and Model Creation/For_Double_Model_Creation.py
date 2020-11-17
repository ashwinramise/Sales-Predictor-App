import plotly.graph_objects as _go
from plotly.subplots import make_subplots as _make_subplots
import numpy as _np
import pandas as _pd
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
from datetime import datetime as _datetime
from sklearn.model_selection import train_test_split as _tts
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn import tree as _tree
from sklearn.model_selection import cross_val_score as _cross_val_score
import matplotlib.pyplot as _plt
import warnings as _warnings
from sklearn.externals import joblib as _joblib
import json


def definedicts(filepath):
    with open(filepath, 'r') as file:
        NewD = json.load(file)
    return NewD

def writecols(df,state,county,brand,subbrand):
    df['State'] = State.get(state)
    df['County'] = County.get(county)
    df['Months'] = [date.month for date in df['Bill Date'].to_list()]
    df['Year'] = [date.year for date in df['Bill Date'].to_list()]
    df['Brand'] = Brand.get(brand)
    df['Sub Brand']= SubBrand.get(subbrand)
    return df


County = definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\Counties.json')
State = definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\States.json')
Brand = definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\Brands.json')
SubBrand = definedicts('.\DataModels\DICTIONARIES_DO_NOT_DEL\SubBrands.json')


def datetimeCleanup(df):
    Datalist=df['Bill Date'].to_list()
    Cleaned = []
    for date in Datalist:
        if '/' in date:
            try:
                p = datetime.strptime(date,'%m/%d/%Y')
            except ValueError:
                try:
                    p = datetime.strptime(date,'%d/%m/%Y')
                except ValueError:
                    p = datetime.strptime(date,'%m/%d/%y')
        elif '-' in date:
            p = datetime.strptime(date,'%m-%d-%Y')
        Cleaned.append(p)
    df['Bill Date'] = Cleaned
    return df

def modelselection(state):
    '''Define as _model_SAH,_model_EM,_PanSales,_NHSales = modelselection(state)'''
    _model_SAH = _joblib.load(f'.\DataModels\State\Predict Orders\{state}\model_SAH.joblib')
    _model_EM = _joblib.load(f'.\DataModels\State\Predict Orders\{state}\model_EM.joblib')
    _PanSales = _joblib.load(f'.\DataModels\State\Predict Sales\{state}\PanSales.joblib')
    _NHSales = _joblib.load(f'.\DataModels\State\Predict Sales\{state}\History_Sales.joblib')
        
    return _model_SAH,_model_EM,_PanSales,_NHSales

_warnings.filterwarnings('ignore')


def create_pars_for_model(merged, model_type='Pandemic', predict='Predict_Sales'):
    """1. merged dataset on dates for the state is required
    2. model_type = 'Pandemic' or 'Natural History'
    'Natural History' Will use the dataset from 2018, 2019 to predict sales based on the demographics.
    'Pandemic' Will use the 2020 DataSet to predict orders and sales
    3. predict = 'Predict_Orders' or 'Predict_Sales'
        if predict = 'Predict_Orders',
        returns - x, ylist ---> List of y = ['R_Res', 'B_Res',
                 'S_Res', 'G_Res', 'NE_Res', 'SAH', 'EM', 'FM', 'Q_Res'] in that order.
        if predict = 'Predict_Sales',
        returns x, y
    3. Returns Variables for model. Can be used as input to model_creation(), run_cross_validation_on_trees()"""
    # Convert the data frame into numerical types
    struct_data = merged.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    if model_type == 'Pandemic':
        struct_data = struct_data[struct_data['Year'] == 2020]
        if predict == 'Predict_Orders':
            names = ['R_Res', 'B_Res',
                     'S_Res', 'G_Res', 'NE_Res', 'SAH', 'EM', 'FM', 'Q_Res']
            x = struct_data[
                ['Covid cases', 'age_median',
                 'family_size', 'income_household_median', 'education_college_or_above',
                 'labor_force_participation']]
            ylist = []
            for name in names:
                y = struct_data[name]
                ylist.append(y)
            return x, ylist
        elif predict == 'Predict_Sales':
            x = struct_data[
                ['R_Res', 'B_Res',
                 'S_Res', 'G_Res', 'NE_Res', 'SAH', 'EM', 'FM', 'Q_Res', 'age_median',
                 'family_size', 'income_household_median', 'education_college_or_above',
                 'labor_force_participation']]
            y = struct_data['Qty']
            return x, y
    elif model_type == 'Natural History':
        struct_data = struct_data[struct_data['Year'] != 2020]
        x = struct_data[
            ['age_median',
             'family_size', 'income_household_median', 'education_college_or_above',
             'labor_force_participation']]
        y = struct_data['Qty']
        return x, y

def createNatHistory(state):
    State = Statemodelcreation[Statemodelcreation['State'] == state]
    struct_data = State.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    x = struct_data[
                ['State', 'County', 'Bill Date', 'Months']]
    y = struct_data['Qty']
    model = RandomForestClassifier(n_estimators=10,random_state=42, bootstrap=True)
    model.fit(x,y)
    joblib.dump(model, f'.\DataModels\State\Predict Sales\{state}\History_Sales.joblib')


def createordersjobib(state):
    State = Statemodelcreation[Statemodelcreation['State'] == state]
    struct_data = State.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    names = ['SAH', 'EM']
    x = struct_data[
                    ['State', 'County',
           'Covid cases', 'age_median', 'family_size',
           'income_household_median', 'education_college_or_above',
           'labor_force_participation', 'unemployment_rate']]
    y = []
    for name in names:
        ym = struct_data[name]
        y.append(ym)
    modelSAH = DecisionTreeClassifier()
    modelSAH.fit(x,y[0])
    joblib.dump(modelSAH, f'.\DataModels\State\Predict Orders\{state}\model_SAH.joblib')
    modelEM = DecisionTreeClassifier()
    modelEM.fit(x,y[1])
    joblib.dump(modelEM, f'.\DataModels\State\Predict Orders\{state}\model_EM.joblib')

    
def createjoblib(state):
    State = Statemodelcreation[Statemodelcreation['State'] == state]
    struct_data = State.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    x = struct_data[
                    ['State', 'County', 'Bill Date', 'Covid cases', 'SAH', 'EM',
           'age_median', 'family_size', 'income_household_median',
           'education_college_or_above', 'labor_force_participation',
           'unemployment_rate', 'Months']]
    y = struct_data['Qty']
    model = ExtraTreesRegressor(n_estimators=10,random_state=42, bootstrap=True)
    model.fit(x,y)
    joblib.dump(model, f'.\DataModels\State\Predict Sales\{state}\PanSales.joblib')



def accuracy(x, y):
    '''Returns the accuracy score of the data model based on training data'''
    model = _DecisionTreeClassifier()
    final_model = model.fit(x, y)
    Xtrain, Xtest, Ytrain, Ytest = _tts(x, y, test_size=0.25)
    predictions = final_model.predict(Xtest)
    score = _accuracy_score(Ytest, predictions)
    return score


def predict_result(final_model, modeltype):
    """1. Takes Final model as input
    specify model type -
    1. Natural History model
    2. COVID model"""

    result = final_model.predict([[]])[0]
    return result


def sales_plot(merged, over='Qty'):
    ''' merged is a merged file containing Index(['Bill Date',
    # 'Qty', 'Sale Count', 'state', 'positive', 'Daily Cases'], dtype ='object') '''
    # Create figure with secondary y-axis
    fig = _make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        _go.Bar(x=merged['Bill Date'], y=merged['Daily Cases'], name="COVID Evolution", opacity=0.50),
        secondary_y=True,
    )

    # Add traces
    fig.add_trace(
        _go.Line(x=merged['Bill Date'], y=merged[over], name="Sales Qty"),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text="COVID vs Sales", plot_bgcolor='rgba(0,0,0,0)'
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Dates")

    # Set y-axes titles
    fig.update_yaxes(title_text="Sales Qty", secondary_y=False)
    fig.update_yaxes(title_text="COVID Evolution", secondary_y=True)

    fig.show()


def viz_tree(model):
    '''Exports the tree to a dot file for visualization'''
    _tree.export_graphviz(model, out_file='FinalModel.dot', feature_names=
    ['R_Res', 'B_Res', 'S_Res', 'G_Res', 'NE_Res', 'SAH', 'EM', 'FM',
     'Q_Res', 'age_median', 'family_size', 'income_household_median',
     'education_college_or_above', 'labor_force_participation'], rounded=True, filled=True)


def model_creation(x, y, depth):
    ''' Creating the parameters for the model, Returns the data model and creates a dot file in the root folder '''
    model = _DecisionTreeClassifier(max_depth=depth)
    final_model = model.fit(x, y)
    return final_model


def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    '''Returns cv_scores_mean, cv_scores_std, accuracy_scores = run_cross_validation_on_trees(X, y, tree_depths, cv=5)'''
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = _DecisionTreeClassifier(max_depth=depth)
        cv_scores = _cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = _np.array(cv_scores_mean)
    cv_scores_std = _np.array(cv_scores_std)
    accuracy_scores = _np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores


# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = _plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean - 2 * cv_scores_std, cv_scores_mean + 2 * cv_scores_std, alpha=0.2)
    ylim = _plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()


def create_models_for_orders(x, y):
    ''' Takes inputs x,y
    Define in code as :
    model_R_Res,model_B_Res,model_S_Res,model_G_Res, model_NE_Res,model_SAH, model_EM, model_FM, model_Q_Res = fdm.create_models_for_orders(x,y)'''
    model_R_Res = model_creation(x, y[0], 4)
    model_B_Res = model_creation(x, y[1], 4)
    model_S_Res = model_creation(x, y[2], 4)
    model_G_Res = model_creation(x, y[3], 4)
    model_NE_Res = model_creation(x, y[4], 4)
    model_SAH = model_creation(x, y[5], 4)
    model_EM = model_creation(x, y[6], 4)
    model_FM = model_creation(x, y[7], 4)
    model_Q_Res = model_creation(x, y[8], 4)
    return model_R_Res, model_B_Res, model_S_Res, model_G_Res, model_NE_Res, model_SAH, model_EM, model_FM, model_Q_Res


def comp_plot(df):
    ''' merged is a merged file containing Index(['Bill Date',
    # 'Qty', 'Sale Count', 'state', 'positive', 'Daily Cases'], dtype ='object') '''
    # Create figure with secondary y-axis
    df2 = df.groupby('Bill Date').agg({'Qty': 'sum', 'Pred': 'sum'}).reset_index()
    fig = _make_subplots(specs=[[{"secondary_y": False
                                  }]])

    fig.add_trace(
        _go.Line(x=df2['Bill Date'], y=df2['Qty'], name="Actual Sales"),
        secondary_y=False,
    )

    # Add traces
    fig.add_trace(
        _go.Line(x=df2['Bill Date'], y=df2['Pred'], name="Predictions"),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text="Sales vs Prediction", plot_bgcolor='rgba(0,0,0,0)'
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Dates")

    # Set y-axes titles
    fig.update_yaxes(title_text="Sales Qty", secondary_y=False)

    fig.show()


def iterring_rows(DataFrame, method):
    '''Define the data frame to operate on in df.
    Method = 'Orders_Pan' or 'Sales_Pan' or Sales_Nhist
    Returns: a list
    Define as alist = iterring_rows(df,method)'''
    DataFrame['Bill Date'] = _pd.to_datetime(DataFrame['Bill Date'],dayfirst=True)
    struct_data = DataFrame.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    if method == 'Sales_Pan':
        df = struct_data[['State', 'County', 'Bill Date', 'Covid cases', 'SAH', 'EM',
           'age_median', 'family_size', 'income_household_median',
           'education_college_or_above', 'labor_force_participation',
           'unemployment_rate', 'Months']]
        df.columns = ['State', 'County', 'Bill_Date', 'Covid_cases', 'SAH', 'EM',
           'age_median', 'family_size', 'income_household_median',
           'education_college_or_above', 'labor_force_participation',
           'unemployment_rate', 'Months']
        # Create an empty list
        Row_list = []
        # Iterate over each row
        for index, rows in df.iterrows():
            # Create list for the current row
            my_list = [rows.State, rows.County, rows.Bill_Date, rows.Covid_cases, rows.SAH, rows.EM,
           rows.age_median, rows.family_size, rows.income_household_median,
           rows.education_college_or_above, rows.labor_force_participation,
           rows.unemployment_rate, rows.Months]

            # append the list to the final list
            Row_list.append(my_list)
    elif method == 'Orders_Pan':
        df = struct_data[['State', 'County',
       'Covid cases', 'age_median', 'family_size',
       'income_household_median', 'education_college_or_above',
       'labor_force_participation', 'unemployment_rate']]
        df.columns = ['State', 'County',
       'Covid_cases', 'age_median', 'family_size',
       'income_household_median', 'education_college_or_above',
       'labor_force_participation', 'unemployment_rate']
        # Create an empty list
        Row_list = []

        # Iterate over each row
        for index, rows in df.iterrows():
            # Create list for the current row
            my_list = [rows.State, rows.County,
       rows.Covid_cases, rows.age_median, rows.family_size,
       rows.income_household_median, rows.education_college_or_above,
       rows.labor_force_participation, rows.unemployment_rate]

            # append the list to the final list
            Row_list.append(my_list)
    elif method == 'Sales_Nhist':
        df = struct_data[['State', 'County', 'Bill Date', 'Months']]
        df.columns =['State', 'County', 'Bill_Date', 'Months']
        # Create an empty list
        Row_list = []

        # Iterate over each row
        for index, rows in df.iterrows():
            # Create list for the current row
            my_list = [rows.State, rows.County, rows.Bill_Date, rows.Months]

            # append the list to the final list
            Row_list.append(my_list)
    else:
        print("Error: Enter correct method: 'Orders_Pan' or 'Sales_Pan' or Sales_Nhist")

    return Row_list


def update_predictions(df, method, state):
    '''
    Creates all required rows and iters over them to create appropriate model.
    Pulls all required models only as and when required.
    Provide the main Data frame from which prediction happens
    method = 'Orders_Pan' or 'Sales_Pan' or 'Sales_Nhist'
    Provide Models for every parameter to be predicted
    Input as df = update_predictions(df, 'Orders_Pan', 'California')
    '''
    df['State'] = StateDict.get(state)
    rows = iterring_rows(df, method)
    _model_SAH,_model_EM,_PanSales,_NHSales = modelselection(state)
    if method == 'Orders_Pan':
        model_SA = []
        model_E = []
        for row in rows:
            SAH =_model_SAH.predict([row])
            EM = _model_EM.predict([row])
            model_SA.append(SAH[0])
            model_E.append(EM[0])
        df['SAH'] = model_SA
        df['EM'] = model_E
        return df
    if method == 'Sales_Pan':
        salelist = []
        for row in rows:
            sale = _PanSales.predict([row])
            salelist.append(sale[0])
        df['Sales Pan'] = salelist
        return df
    if method == 'Sales_Nhist':
        salelist = []
        for row in rows:
            sale = _NHSales.predict([row])
            salelist.append(sale[0])
        df['Natural Sales'] = salelist
        return df
    
    
## An Alternate trial Model ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>> RefOnly  
def Alternate(df,depth=4):
    struct_data = df.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    x = struct_data[
                ['Covid cases', 'age_median',
                 'family_size', 'income_household_median', 'education_college_or_above',
                 'labor_force_participation']]
    y = struct_data['Qty']
    model = _DecisionTreeClassifier(max_depth=depth)
    final_model = model.fit(x, y)
    return final_model

# Iterrowcreation and prediction
def prediction(dataf, final_model):
    df = dataf[['Covid cases', 'age_median',
                 'family_size', 'income_household_median', 'education_college_or_above',
                 'labor_force_participation']]
    df.columns = ['Covid_cases','age_median', 'family_size', 'income_household_median',
                      'education_college_or_above', 'labor_force_participation']
    struct_data = df.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[_np.number]).columns)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in non_numeric_columns:
        if col in struct_data.columns:
            i = struct_data.columns.get_loc(col)
            struct_data.iloc[:, i] = struct_data.apply(lambda i: le.fit_transform(i.astype(str)), axis=0,
                                                       result_type='expand')
    # Create an empty list
    Row_list = []
    # Iterate over each row
    for index, rows in struct_data.iterrows():
        # Create list for the current row
        my_list = [rows.Covid_cases, rows.age_median, rows.family_size, rows.income_household_median,
                   rows.education_college_or_above, rows.labor_force_participation]
        Row_list.append(my_list)
    salelist = []
    for row in Row_list:
        sale = final_model.predict([row])
        salelist.append(sale[0])
    dataf['Sales Pan'] = salelist
    return dataf

