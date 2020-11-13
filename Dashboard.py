import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from datetime import datetime
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64
import warnings
import io
import Predictor as pr
import json

warnings.filterwarnings('ignore')

# ------------------------------------REQUIRED FILES IMPORT---------------------------------------------------------
# Global-sets
E_Data = pd.read_csv('.\DataModels\DataSets DO_NOT_DEL\Sales,COVID,Demogrphics,Types.csv')
# E_Data['Bill Date'] = pd.to_datetime(E_Data['Bill Date'],format='%m/%d/%y')
E_Data = pr.dateconvert(E_Data)
Datewise = E_Data.groupby(['State', 'County', 'Year', 'Bill Date']).agg(
    {'Covid cases': 'first', 'Qty': 'sum'}).reset_index()
Brandwise = E_Data.groupby(['State', 'County', 'Year', 'Bill Date', 'Brand', 'Sub Brand']).agg(
    {'Qty': 'sum'}).reset_index()
Bystate = Datewise.groupby(['State', 'Bill Date']).agg({'Covid cases': 'sum', 'Qty': 'sum'}).reset_index()
Byyear = Datewise.groupby(['State', 'Year', 'Bill Date']).agg({'Covid cases': 'sum', 'Qty': 'sum'}).reset_index()
Bystate_Brand = Brandwise.groupby(['State', 'Brand', 'Bill Date']).agg({'Qty': 'sum'}).reset_index()

# ------------------------------------REQUIRED FILES IMPORT---------------------------------------------------------
# Images
Table = r'.\Images\Data_Structure.png'
table_base64 = base64.b64encode(open(Table, 'rb').read()).decode('ascii')
graph = r'.\Images\California_LosAngeles(Nhist96.1,Pan96.4).png'
graph_base64 = base64.b64encode(open(graph, 'rb').read()).decode('ascii')

keys = [E_Data['State'].unique()]

# ------------------------------------REQUIRED FUNCTIONS WRITTEN---------------------------------------------------------

# Creates a list of dictionaries, which have the keys 'label' and 'value'.
def get_options(list_states):
    dict_list = []
    for i in list_states:
        dict_list.append({'label': i, 'value': i})

    return dict_list


#  create a column of cumulative cases and return a dataframe
def countydf_creation(state, county=None, method='C', brand=None, subbrand=None, year=None):
    ''' 1. Enter State
    2. Enter County
    3. Enter method: Datewise , Brandwise '''
    global df_by_state
    if method == 'C' or method == 'S':
        if year is None:
            df_by_state = Datewise[(Datewise['State'] == state) & (Datewise['County'] == county)]
        elif year is None and county is None:
            df_by_state = Bystate[Bystate['State'] == state]
        elif county is None:
            df_by_state = Datewise[
                (Datewise['State'] == state) & (Datewise['Year'] == year)]
        else:
            df_by_state = Datewise[
                (Datewise['State'] == state) & (Datewise['County'] == county) & (Datewise['Year'] == year)]
    elif method == 'B':
        if subbrand is None:
            df_by_state = Brandwise[(Brandwise['State'] == state) & (Brandwise['County'] == county) &
                                    (Brandwise['Brand'] == brand) & (Brandwise['Year'] == year)]
        elif year is None and subbrand is None:
            df_by_state = Brandwise[(Brandwise['State'] == state) & (Brandwise['County'] == county) &
                                    (Brandwise['Brand'] == brand)]
        elif year is None and county is None:
            df_by_state = Bystate_Brand[(Bystate_Brand['State'] == state) & (Bystate_Brand['Brand'] == brand)]
        else:
            df_by_state = Brandwise[(Brandwise['State'] == state) & (Brandwise['County'] == county) &
                                    (Brandwise['Brand'] == brand) & (Brandwise['Sub Brand'] == subbrand) &
                                    (Brandwise['Year'] == year)]
    return df_by_state
# ------------------------------------REQUIRED FUNCTIONS WRITTEN---------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#111111',
    # 'background': '#e8dfdf',
    'text': 'white'
}

app.layout = html.Div(style={'background': '#333333',
                             'width': '100%',
                             'height': '100%',
                             'borderRadius': '5px',
                             },
                      children=[
                          html.Div(
                              [html.A(
                                  [html.Div([
                                      html.Img(src='https://brand.ncsu.edu/img/logo/brick2x2.jpg', height="70px",
                                               style={'textAlign': 'left'}
                                               ),
                                  ], className="one-third column", style={'textAlign': 'left'}),
                                      html.Div(
                                          [html.H2(className='About', children='SALES PREDICTION AND ANALYSIS TOOL',
                                                   style={'textAlign': 'center',
                                                          'font-size': '20pt', 'color': 'yellow',
                                                          'font-weight': 'bold'})
                                           ],
                                          className="one-half column",
                                          id="title",
                                          style={'textAlign': 'center'}
                                      ),
                                      html.Div([html.Img(
                                          src='https://www.ise.ncsu.edu/wp-content/uploads/2016/01/ise-symbol-logo-01.jpg',
                                          height="70px", )
                                      ], className="one-third column", style={'textAlign': 'right'})
                                  ], style={'display': 'flex', 'width': '100%'}, href="https://www.ncsu.edu/",
                              ),
                              ], style={'background': '#404040', 'color': 'black', 'width': '100%'}),

                          html.Div(id='Selection_Tabs', className='control-tabs', children=[
                              dcc.Tabs(id='All_Tabs', value='All', parent_className='custom-tabs',
                                       className='custom-tabs-container',
                                       children=[
                                           dcc.Tab(
                                               label='Introduction',
                                               value='Introduction',
                                               className='custom-tab',
                                               selected_className='custom-tab--selected',
                                               children=[
                                                   html.Div(
                                                       className="Nesecity Explained",
                                                       children=[
                                                           html.H1(className='About', children='Problem Definition',
                                                                   style={'textAlign': 'center', 'font-size': '15pt',
                                                                          'font-weight': 'bold'}),
                                                           html.H4(className='About', children='Introduction',
                                                                   style={'textAlign': 'left', 'font-size': '15pt',
                                                                          'font-weight': 'bold'}),
                                                           html.P(
                                                               'The Coronavirus (COVID-19) pandemic has changed life for everyone in '
                                                               'unprecedented ways and is arguably one of the '
                                                               'most difficult challenges faced in modern history. It has affected '
                                                               'every industry to varying degrees and predicting the future from hereon '
                                                               'is an issue that everybody is grappling with. PC and laptop '
                                                               'manufacturers have reported declining quarterly sales globally as a '
                                                               'rapid surge in demand for notebooks is being witnessed by all major '
                                                               'brands in the market like Apple, HP, Lenovo, and Asus. The pandemic has '
                                                               'given rise to such novel market scenarios, and companies are keen to '
                                                               'streamline their supply chain to incorporate these variations and '
                                                               'ensure steady profits. Lenovo Group of Technologies being one of the '
                                                               'front runners in the computer market also wants to find optimal '
                                                               'production strategies for their various products and customer type.'),
                                                           html.P(
                                                               'Lenovo has enjoyed a global presence and even in the current pandemic '
                                                               'scenario, it has improved its market share to 24.1%. However, '
                                                               'with the countrywide lockdowns followed by state and county policies '
                                                               'established due to COVID-19, Lenovo faced disruption in its supply '
                                                               'chain. Moreover, Wuhan, the epicenter of the COVID-19 outbreak, '
                                                               'is also the major manufacturing location of Lenovo which started facing '
                                                               'production restrictions as early as January 2020. The causal effect of '
                                                               'these manufacturing restrictions on the sales of Lenovo products in the '
                                                               'United States will be studied through this project.'),
                                                           html.H4(className='About', children='Objective',
                                                                   style={'textAlign': 'left', 'font-size': '15pt',
                                                                          'font-weight': 'bold'}),
                                                           html.P(
                                                               'This project aims to identify factors that influence the demand for '
                                                               'Lenovo products concerning the executive orders '
                                                               'in place due to the COVID -19 pandemic and study their correlation. Key '
                                                               'variables will be identified to build a mathematical model to '
                                                               'understand consumer behavior and predict future demand in the wake of a '
                                                               'similar situation in the future. The project will assess if individual '
                                                               'state responses influence the sales of Lenovo products and aid in '
                                                               'deciding optimal production strategies based on customer type and '
                                                               'product type as a function of geographic region, industry, and pandemic '
                                                               'response. Customer type will be defined in terms of the quantity of '
                                                               'sales units.'),
                                                       ],
                                                       style={
                                                           'textAlign': 'left', 'font-size': '15pt',
                                                           'background': '#ffffcc'})

                                               ], style={"background": "#80aaff", 'color': 'black'}
                                           ),
                                           dcc.Tab(
                                               label='Instructions For Use',
                                               value='about',
                                               className='custom-tab',
                                               selected_className='custom-tab--selected',
                                               children=[html.Div([
                                                   html.Div(
                                                       className="About The Tool",
                                                       children=[
                                                           html.H4(className='About', children='About The Application',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.P('The Purpose of Building this application was to'
                                                                  ' provide an intutive tool that could easily be '
                                                                  'deployed on any system without having to install '
                                                                  'too many special applications. The application is'
                                                                  ' built on PLOTLY Dash which is based on interfacing '
                                                                  'Python and HTML'),
                                                           html.H4(className='About', children='Software Requirements',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.P('The advantage of using Dash is that the app can be '
                                                                  'deployed on any system that runs Python'),
                                                           html.P('System requires to have a web browser (Preferably Chrome based)'
                                                                  'and Python 3.7+ with updated versions of dash-1.16.3+'
                                                                  ', pandas-0.25.3+, numpy-1.19.2+'),
                                                           html.H4(className='About', children='The Tool and how to use it:',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.P('The Data Visualization Tab allows the '
                                                                  'visualization of COVID Data / Sales Data and '
                                                                  'Brandwise Sales Data provided to us by Lenovo. '
                                                                  'The Dataset has been defined '
                                                                  'in the code directly. It can be modified to '
                                                                  'accepting input if required.'),
                                                           html.H4(className='About', children='Prediction of sales',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.P('For the prediction of sales, and Extreme Trees Regressor'
                                                                  ' machine learning algorithm from Sci-Kit learn'
                                                                  ' has been used. The reason ETR was chosen was because'
                                                                  ' of its faster execution time in comparison to a '
                                                                  'Random Forrest regressor which is more widely known '
                                                                  'while maintaining a comparable accuracy'),
                                                           html.P('The "Prediction of Sales" tab accepts a file as '
                                                                  'input and predicts the sales based on the various '
                                                                  'parameters.'),
                                                           html.H6(className='About', children='File Type and Data Structure:',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.P('The input file should be composed in the following format'),
                                                           html.Img(src='data:image/png;base64,{}'.format(table_base64),
                                                                    height="400px"),
                                                           html.P('Once the file path is defined, hit submit to upload'),
                                                           html.P('1. A Natural History Model  '),
                                                           html.P('2. A Pandemic Based Model'),
                                                           html.P('Depending on which model is '
                                                                  'chosen, the relevant classifications will be done '
                                                                  'and appropriate results provided. '),
                                                           html.H6(className='About', children='Data Model Test Results',
                                                                   style={'textAlign': 'left', 'font-weight': 'bold'}),
                                                           html.Img(src='data:image/png;base64,{}'.format(graph_base64),
                                                                                                                height="500px",)
                                                       ],
                                                       style={
                                                           'textAlign': 'left', 'font-size': '15pt', 'color': 'black',
                                                           'background': '#ffffcc'}),],
                                                   style={'textAlign': 'left', 'background': '#ffffcc'}),

                                               ], style={"background": "#80aaff", 'color': 'black'}),

                                           dcc.Tab(
                                               label='Data Visualisation',
                                               value='DV',
                                               className='custom-tab',
                                               selected_className='custom-tab--selected',
                                               children=[html.Div([
                                                   # html.Div(
                                                   #     className="row header",
                                                   #     children=[
                                                   #         html.H4('Data Visualisation For COVID and Sales',
                                                   #                 style={
                                                   #                     'textAlign': 'center'})
                                                   #     ],
                                                   # ),

                                                   html.Div([
                                                       html.Div(className='div-for-dropdown-covid/sales',
                                                                children=[
                                                                    dcc.Dropdown(id='goverselector',
                                                                                 options=[
                                                                                     {'label': 'COVID', 'value': 'C'},
                                                                                     {'label': 'Sales', 'value': 'S'},
                                                                                     {'label': 'Brand', 'value': 'B'},
                                                                                 ],
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='goverselector',
                                                                                 placeholder='Select Group')],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}),

                                                       html.Div(className='div-for-dropdown-year',
                                                                children=[
                                                                    dcc.Dropdown(id='yearSelect',
                                                                                 options=[
                                                                                     {'label': '2018', 'value': 2018},
                                                                                     {'label': '2019', 'value': 2019},
                                                                                     {'label': '2020', 'value': 2020},
                                                                                 ],
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='yearSelect',
                                                                                 placeholder='Select Year')],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}),

                                                       html.Div(className='div-for-dropdown Brand',
                                                                children=[
                                                                    dcc.Dropdown(id='BrandSelector',
                                                                                 options=get_options(
                                                                                     E_Data['Brand'].unique()),
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='BrandSelector',
                                                                                 placeholder='Select Brand ( If Brand Option )',
                                                                                 searchable=True
                                                                                 )],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}
                                                                ),
                                                       html.Div(className='div-for-dropdown Subbrands',
                                                                children=[
                                                                    dcc.Dropdown(id='subBrandselector',
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='subBrandselector',
                                                                                 placeholder='Select Sub Brand ( If Brand Option )',
                                                                                 searchable=True
                                                                                 )],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}
                                                                )],
                                                       style={'display': 'flex', 'width': '100%'}),

                                                   html.Div([

                                                       html.Div(className='div-for-dropdown',
                                                                children=[
                                                                    dcc.Dropdown(id='StateSelector',
                                                                                 options=get_options(
                                                                                     E_Data['State'].unique()),
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='StateSelector',
                                                                                 placeholder='Select State')
                                                                ],
                                                                style={'color': 'black', 'width': '50%'}),

                                                       html.Div(className='div-for-dropdown 2',
                                                                children=[
                                                                    dcc.Dropdown(id='CountySelector',
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='CountySelector',
                                                                                 placeholder='Select County',
                                                                                 searchable=True
                                                                                 )
                                                                ],
                                                                style={'color': 'black', 'width': '50%'})
                                                   ],
                                                       style={'display': 'flex', 'width': '100%'}),

                                                   html.Div(['Select Date Range',  # add DateRangePicker here
                                                             dcc.DatePickerRange(
                                                                 id='date-input',
                                                                 stay_open_on_select=False,
                                                                 min_date_allowed=datetime(2018, 1, 1),
                                                                 max_date_allowed=datetime(2020, 8, 31),
                                                                 initial_visible_month=datetime.now(),
                                                                 start_date=datetime(2018, 1, 1),
                                                                 end_date=datetime(2020, 8, 31),
                                                                 number_of_months_shown=2,
                                                                 month_format='MMMM,YYYY',
                                                                 display_format='YYYY-MM-DD',
                                                                 style={
                                                                     # 'color': '#120101',
                                                                     'font-size': '15px',
                                                                     'width': '50%'
                                                                 }
                                                             ),

                                                             html.Div(id='date-output')
                                                             ], className="row ",
                                                            style={'marginTop': 30, 'marginBottom': 0, 'font-size': 18,
                                                                   'color': 'white',
                                                                   "background": "#99ceff"}),
                                                   html.Div(id='graph-output')

                                               ], style={"background": "#80aaff", 'font': 'black'})
                                               ], style={"background": "#80aaff", 'font': 'black'}),

                                           dcc.Tab(
                                               label='Prediction of sales',
                                               value='predict',
                                               className='custom-tab',
                                               selected_className='custom-tab--selected',
                                               children=[
                                                   html.Div([
                                                       dcc.Input(id='filepath', type='text', placeholder="Enter Path for file ending with *.csv",
                                                                 style={'width': '50%'}),
                                                       html.Button(id='submit-button', type='submit', children='Submit'
                                                                   ,style={'background-color':'#008CBA',
                                                                           'font-size': '12px',
                                                                           'color': 'white',
                                                                            'text-align': 'center',

                                                                           })
                                                       ,
                                                       html.Div([
                                                           html.Div(className='div-for-dropdown-Model-Type',
                                                                    children=[
                                                                        dcc.Dropdown(id='modelSelect',
                                                                                     options=[
                                                                                         {'label': 'Natural History',
                                                                                          'value': 'Natural_Hist'},
                                                                                         {'label': 'Pandemic Based',
                                                                                          'value': 'PandemicSales'},
                                                                                     ],
                                                                                     multi=False,
                                                                                     style={'backgroundColor': 'white',
                                                                                            'width': '100%',
                                                                                            'height': '15px'
                                                                                            },
                                                                                     className='modelSelect',
                                                                                     placeholder='Select Model Type')],
                                                                    style={
                                                                        'marginBottom': 30,
                                                                           'font-size': 18,
                                                                           'color': 'black',
                                                                           'width': '25%'}),
                                                           html.Div(className='div-for-dropdown-3',
                                                                    children=[
                                                                        dcc.Dropdown(id='State2Selector',
                                                                                     options=get_options(
                                                                                         E_Data['State'].unique()),
                                                                                     multi=False,
                                                                                     style={'backgroundColor': 'white',
                                                                                            'width': '100%',
                                                                                            'height': '15px'
                                                                                            },
                                                                                     className='State2Selector',
                                                                                     placeholder='Select State Specific'
                                                                                                 ' Model')
                                                                    ],
                                                                    style={'marginBottom': 30,
                                                                           'font-size': 18,
                                                                           'color': 'black',
                                                                           'width': '25%'}),
                                                                html.Div(className='div-for-dropdown 2',
                                                                children=[
                                                                    dcc.Dropdown(id='County2Selector',
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='County2Selector',
                                                                                 placeholder='Select County',
                                                                                 searchable=True
                                                                                 )
                                                                ],
                                                                style={'color': 'black', 'width': '25%'}),
                                                        html.Div(className='div-for-dropdown 3',
                                                                children=[
                                                           dcc.Dropdown(id='pieselect',
                                                                        options=[
                                                                            {'label': 'Laptop/Desktop/Workstation',
                                                                             'value': 'LDW'},
                                                                            {'label': 'Consumer/Commercial/Gaming',
                                                                             'value': 'CCG'},
                                                                            {'label': 'Brands',
                                                                             'value': 'BrandQuantities'},
                                                                            {'label': 'Geographical region',
                                                                             'value': 'Geographical'}
                                                                        ],
                                                                        multi=False,
                                                                        style={'backgroundColor': 'white',
                                                                               'width': '100%',
                                                                               'height': '15px'
                                                                               },
                                                                        className='pieselect',
                                                                        placeholder='Select Classification Type')],
                                                           style={'marginBottom': 30,
                                                                  'font-size': 18,
                                                                  'color': 'black',
                                                                  'width': '25%'}),

                                                       ], style={'display': 'flex', 'width': '100%','marginTop':10}),

                                                       html.Div([
                                                            html.Div(className='div-for-dropdown Brand',
                                                                children=[
                                                                    dcc.Dropdown(id='Brand2Selector',
                                                                                 options=get_options(
                                                                                     E_Data['Brand'].unique()),
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='Brand2Selector',
                                                                                 placeholder='Select Brand ( If Brand Option )',
                                                                                 searchable=True
                                                                                 )],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}
                                                                ),
                                                       html.Div(className='div-for-dropdown Sub-brands',
                                                                children=[
                                                                    dcc.Dropdown(id='subBrand2selector',
                                                                                 multi=False,
                                                                                 style={'backgroundColor': 'white',
                                                                                        'width': '100%',
                                                                                        'height': '15px'
                                                                                        },
                                                                                 className='subBrand2selector',
                                                                                 placeholder='Select Sub Brand ( If Brand Option )',
                                                                                 searchable=True
                                                                                 )],
                                                                style={'marginTop': 10, 'marginBottom': 30,
                                                                       'font-size': 18,
                                                                       'color': 'black',
                                                                       'width': '25%'}
                                                                )], style={'display': 'flex', 'width': '100%'}
                                                       ),
                                                       html.Div(
                                                           [html.Div(id='prediction-output',
                                                                     className="one-half column",
                                                                     style={'textAlign': 'center'}
                                                                     ),
                                                            html.Div(id='picharts',
                                                                     className="one-half column",
                                                                     style={'textAlign': 'center'}
                                                                     )
                                                            ]
                                                       )

                                                   ], style={'marginTop': 10, 'marginBottom': 10, 'font-size': 18,
                                                             'color': 'grey',
                                                             'width': '99%'}),


                                               ], style={"background": "#80aaff", 'font': 'grey'})
                                       ])

                          ], style={"background": "80aaff", 'font': 'yellow'}),

                          html.Div(
                                   children=[
                                       html.Img(
                                           src='https://wallpapercave.com/wp/wp710917.png',
                                           style={'height': '100%', 'width':'100%', 'textAlign': 'left'})
                                   ], style={'textAlign': 'left'}
                                   ),
                      ])

# Dictionary for Relative dropdown SubBrand Selection
subBrands = {}
for brand in E_Data['Brand'].unique():
    df = E_Data[E_Data['Brand'] == brand]
    subbrands = []
    for sub in df['Sub Brand'].unique():
        subbrands.append(sub)
    subBrands.update({brand: subbrands})


@app.callback(
    Output('subBrandselector', 'options'),
    [Input('BrandSelector', 'value')])
def set_subbrand_options(selected_state):
    return [{'label': i, 'value': i} for i in subBrands[selected_state]]


@app.callback(
    Output('subBrand2selector', 'options'),
    [Input('Brand2Selector', 'value')])
def set_subbrand_options(selected_state):
    return [{'label': i, 'value': i} for i in subBrands[selected_state]]

# Dictionary for Relative dropdown SubBrand Selection


# Dictionary for Relative dropdown County Selection
State_Counties = {}
for state in E_Data['State'].unique():
    df = E_Data[E_Data['State'] == state]
    counties = []
    for county in df['County'].unique():
        counties.append(county)
    State_Counties.update({state: counties})


@app.callback(
    Output('CountySelector', 'options'),
    [Input('StateSelector', 'value')])
def set_county_options(selected_state):
    return [{'label': i, 'value': i} for i in State_Counties[selected_state]]

@app.callback(
    Output('County2Selector', 'options'),
    [Input('State2Selector', 'value')])
def set_county_options(selected_state):
    return [{'label': i, 'value': i} for i in State_Counties[selected_state]]


# Dictionary for Relative dropdown County Selection


@app.callback(Output('graph-output', 'children'),
              [Input('date-input', 'start_date'),  # Add start date
               Input('date-input', 'end_date'),
               Input('goverselector', 'value'),
               Input('StateSelector', 'value'),
               Input('CountySelector', 'value'),
               Input('BrandSelector', 'value'),
               Input('subBrandselector', 'value'),
               Input('yearSelect', 'value')
               ])  # Add end date
def render_graph(start_date, end_date, condition, state, county, brand, subbrand, year):
    df_by_state = countydf_creation(state, county, condition, brand, subbrand, year)
    data = df_by_state[(df_by_state['Bill Date'] >= start_date) & (df_by_state['Bill Date'] <= end_date)]
    if condition == 'C':
        return dcc.Graph(
            id='graph-1',
            figure={
                'data': [
                    {'x': data['Bill Date'], 'y': data['Covid cases'], 'type': 'bar', 'name': 'value1'},
                ],
                'layout': {
                    'title': f'{state},{county} | Cumulative COVID Cases',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text'],
                        'size': 18
                    },
                    'xaxis': {
                        'title': 'Date',
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor',
                    },
                    'yaxis': {
                        'title': 'COVID',
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor'
                    },

                }
            }
        )
    elif condition == 'S':
        return dcc.Graph(
            id='graph-1',
            figure={
                'data': [
                    {'x': data['Bill Date'], 'y': data['Qty'], 'type': 'bar', 'name': 'value1'},
                ],
                'layout': {
                    'title': f'{state},{county} daily Sales',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text'],
                        'size': 18
                    },
                    'xaxis': {
                        'title': 'Date',
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor',
                    },
                    'yaxis': {
                        'title': 'Sales',
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor'
                    },

                }
            }
        )
    elif condition == 'B':
        return dcc.Graph(
            id='graph-1',
            figure={
                'data': [
                    {'x': data['Bill Date'], 'y': data['Qty'], 'type': 'bar', 'name': 'value1'},
                ],
                'layout': {
                    'title': f'{state},{county} daily Sales \n {brand}, {subbrand}',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text'],
                        'size': 10
                    },
                    'xaxis': {
                        'title': None,
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor',
                    },
                    'yaxis': {
                        'title': 'Sales',
                        'showspikes': True,
                        'spikedash': 'dot',
                        'spikemode': 'across',
                        'spikesnap': 'cursor'
                    },

                }
            }
        )


# MainChart for Predictions
@app.callback(Output('prediction-output', 'children'),
              [Input('State2Selector', 'value'),
               Input('County2Selector', 'value'),
               Input('modelSelect', 'value'),
               Input('Brand2Selector', 'value'),
               Input('subBrand2selector', 'value'),
               Input('submit-button', 'n_clicks')],
                State('filepath','value'))
def graphing_preds(state,county,method,brand,subbrand,submit,filepath):
    if submit>0:
        df = pr.predictor(pd.read_csv(filepath),state,county,method,brand,subbrand)
        return pr.sales_plot(df,method,brand,subbrand)


# Piechart for classification
@app.callback(Output('picharts','children'),
            [  Input('State2Selector', 'value'),
               Input('County2Selector', 'value'),
               Input('pieselect','value'),
               Input('modelSelect', 'value')])
def update_figure(state,county,method,model):
    if model == 'Natural_Hist':
        Dict = pr._definedicts(f'.\DataModels\DICTIONARIES_DO_NOT_DEL\{method}.json')
        if method != 'Geographical':
            df = pd.DataFrame.from_dict([Dict.get(state).get(county)]).T
        else:
            df = pd.DataFrame.from_dict([Dict.get(state)]).T
        fig = px.pie(df, values=df[0].to_list(), names=df.index.to_list(), title=f'Comparison by {method}')
        fig.show()
    else:
        pass

if __name__ == '__main__':
    app.run_server(debug=True)


