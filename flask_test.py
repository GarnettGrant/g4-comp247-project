from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import joblib
import sys
import traceback

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET','POST'])
def predict():
    #load the model
    lrg = joblib.load('model.pkl')
    # lrg_scaler = joblib.load('scaler.pkl')
    
    if lrg and request.method == 'POST':
        try:
            json_data = request.get_json()
            print("json_data: ", json_data)
            print("request data: ", request.data)
            print("request headers", request.headers)
            
            # Convert scalar values to 1-item lists
            json_data = {key: [value] for key, value in json_data.items()}
            
            #Convert json into df
            query_df = pd.DataFrame(json_data)
            print("query_df: ", query_df)

            predict_list = lrg.predict(query_df).tolist()
            response_data = {"predictions": predict_list}
            
            print("response_data: ", response_data)
            return jsonify(response_data)
            
            # return ("Done!")
        except:
            return jsonify({'trace': traceback.format_exc()})
    elif request.method == 'GET':
        return render_template('index.html')
    else:
        print('Train the model first')
        return ('No trained model') 
    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)