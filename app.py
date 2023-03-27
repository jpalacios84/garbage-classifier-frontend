from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_gif_component as gif
import flask
from flask import Flask, Response
from webcam_support import gen, VideoCamera
import os, cv2
import numpy as np

import tensorflow as tf
model = tf.keras.models.load_model('./assets/CNN_3Class_Best.h5')
predict_dict = {0: 'Compost', 1: 'Recyclable', 2: 'Trash'}

server = Flask(__name__)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, './assets/style.css'], server=server)

app.layout = html.Div(children=[
    dbc.Row([
        dbc.Col(gif.GifPlayer(gif='assets/logo_small.gif', still='assets/logo_small.gif'), width='auto')
    ], justify='center'),
    dbc.Row([
        dbc.Card(
        [            
            dbc.CardBody(
                [
                    html.H4('Garbage classifier', className="card-title"),                    
                    dbc.Row([                        
                            dbc.Col(html.P('Point the camera at the object you want to classify', className='card-text'), width='auto'),
                            dbc.Col(html.P('Unclassified!', id='div-results'), width='auto'),
                        ], justify='between'),
                    dbc.Spinner(dbc.CardImg(id='still-image', src='/video_feed', top=False)),                    
                    dbc.Button('Classify!', style={ 
                                                    'font-size' : 'x-large',
                                                    'background-color' : '#40513B', 
                                                    'border-color' : 'white' , 
                                                    'width' : '100%' 
                                                }, id='button-classify'),
                ]
            ),
        ],
        style={ 'width': '800px' }), 
    ], justify='center'),
])

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.callback(
    Output('div-results', 'children'), 
    Input('button-classify', 'n_clicks')
    )
def classify_from_cam(n_clicks):
    if n_clicks != None:
        vc = VideoCamera()
        success, img = vc.video.read()
        if success:        
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resize = tf.image.resize(img, (256,256))
            yhat = model.predict(np.expand_dims(resize/255, 0))
            return predict_dict[int(np.argmax(yhat, axis = 1))]
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server()