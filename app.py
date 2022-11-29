from flask import Flask,request,jsonify
import numpy as np
import pickle
with open('logit_pkl','rb') as f:
    logit_model = pickle.load(f, encoding='UTF-8')
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cpt = int(request.form.get('cpt'))
    bp = float(request.form.get('bp'))
    chol = float(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    ekg = int(request.form.get('ekg'))
    hr = float(request.form.get('hr'))
    ea = int(request.form.get('ea'))
    st = float(request.form.get('st'))
    sts = int(request.form.get('sts'))
    flo = int(request.form.get('flo'))
    t = int(request.form.get('t'))
    lis=[[age,sex,cpt,bp,chol,fbs,ekg,hr,ea,st,sts,flo,t]]
    print(lis)


    input_query = np.array(lis)
    ans = logit_model.predict(input_query)
    print(ans)
    return jsonify({'Alert':str(ans)})
if __name__ == '__main__':
    app.run(debug=True)