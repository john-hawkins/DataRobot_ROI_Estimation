from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from config import Config
import pandas as pd
import datarobot as dr
import roi   # Import the file: roi.py
import opti  # Optimise over 
import os

app = Flask(__name__)
app.config.from_object(Config)


ALLOWED_EXTENSIONS = set(['csv', 'tsv'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ###################################################################################
# Index Page
@app.route('/')
def index():
    # GET THE LIST OF PROJECTS
    projs = dr.Project.list()
    return render_template("index.html", projects=projs)


# ###################################################################################
# Choose an analysis approach page
@app.route('/approach/<projectId>/')
@app.route('/approach', methods = ['POST', 'GET'])
def approach():
    if request.method == 'POST':
       projectId = request.form["projectId"]
    proj = dr.Project.get(project_id=projectId)
    proj_type = proj.target_type
    mods = proj.get_models()

    if proj_type == 'Binary':
       return render_template("binary.html", project=proj, models=mods)
    if proj_type == 'Regression':
       return render_template("regression.html", project=proj, models=mods)
   
    projs = dr.Project.list()
    return render_template("unsupported.html", project=proj, models=mods, projects=projs)

# ###################################################################################
# Discrete Boundary between useful and costly predictions 
@app.route('/regression_discrete', methods = ['POST', 'GET'])
def regression_discrete():
    return render_template("not_implemented_yet.html")

# ###################################################################################
# Continuous relatiponsip between error and costs 
@app.route('/regression_continuous', methods = ['POST', 'GET'])
def regression_continuous():
    return render_template("not_implemented_yet.html")

# ###################################################################################
# Costs are proportional to Aggregated Error
@app.route('/regression_aggregate', methods = ['POST', 'GET'])
def regression_aggregate():
    return render_template("not_implemented_yet.html")


# ###################################################################################
# Cost/Benefit Payoff Analysis For Binary Classification
@app.route('/costbenefit', methods = ['POST', 'GET'])
def costbenefit():
    if request.method == 'POST':
       projectId = request.form["projectId"]
       if 'tp' in request.values:
          tp = float(request.form["tp"])
          fp = float(request.form["fp"])
          tn = float(request.form["tn"])
          fn = float(request.form["fn"])
          cases = float(request.form["cases"])
          baserate = float(request.form["baserate"])
          num_models = int(request.form["num_models"])
       else:
          num_models = 1
          tp = 1000
          fp = -200
          tn = 0
          fn = 0
          cases = 1000 
          baserate = 0.01
    else:
       projectId = request.args.get("projectId")

    if projectId == None:
       return render_template("error.html")
    else:
       proj = dr.Project.get(project_id=projectId)
       mods = proj.get_models()
       modroi = roi.evalBinaryClassModels(project=proj, models=mods, num_models=num_models, 
                                         tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, baserate=baserate)

       return render_template("costbenefit.html", 
                               project=proj, 
                               models=modroi, num_models=num_models,
                               tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, baserate=baserate)


# ########################################################################################
# Intervention Style Analysis for Binary Classification
@app.route('/intervention', methods = ['POST', 'GET'])
def intervention():
    if request.method == 'POST':
       projectId = request.form["projectId"]
    else:
       projectId = request.args.get("projectId")

    if 'payoff' in request.values:
       num_models=int(request.form["num_models"])
       cases = float(request.form["cases"])
       cost = float(request.form["cost"])
       baserate = float(request.form["baserate"])
       succrate = float(request.form["succrate"])
       backfire = float(request.form["backfire"])
       payoff = float(request.form["payoff"])
       payback = float(request.form["payback"])
    else:
       cases = 1000
       cost  = 10
       baserate = 0.01
       succrate = 0.1
       backfire = 0.03
       payoff = 1000
       payback = -400
       num_models=1
    print(cases)

    if projectId == None:
       return render_template("error.html")
    else:
       proj = dr.Project.get(project_id=projectId)
       mods = proj.get_models()
       tp, fp, tn, fn = roi.convertIntervention(cases=cases, baserate=baserate, 
                                                cost=cost, payoff=payoff, payback=payback, 
                                                succrate=succrate, backfire=backfire)
       modroi = roi.evalBinaryClassModels(project=proj, models=mods, num_models=num_models,
                                         tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, baserate=baserate)

       return render_template("intervention.html",
                               project=proj,
                               models=modroi, num_models=num_models,
                               cases=cases, baserate=baserate, cost=cost, payoff=payoff, payback=payback,
                               succrate=succrate, backfire=backfire)

# ########################################################################################
# Optimization for Binary Classification
@app.route('/optimization', methods = ['POST', 'GET'])
def optimization():
    if request.method == 'POST':
       projectId = request.form["projectId"]
    else:
       projectId = request.args.get("projectId")

    if projectId == None:
       return render_template("error.html")
    else:
       proj = dr.Project.get(project_id=projectId)
       mods = proj.get_models()
       mod = mods[0]
       featurelist_id = mod.featurelist_id
       feats = mod.get_features_used()
       fList = dr.Featurelist.get(projectId, featurelist_id)
       features = fList.features
       return render_template("optimization.html", project=proj, models=mods, mod=mod, features=feats)



# ########################################################################################
# Run the Optimization for Binary Classification
@app.route('/runoptimization', methods = ['POST', 'GET'])
def runoptimization():
    print("/runoptimization :", request.method)

    if request.method == 'POST':
        projectId = request.form["projectId"]
        modelId = request.form["modelId"]
        colOne = request.form["colone"]
        colTwo = request.form["coltwo"]

        proj = dr.Project.get(project_id=projectId)
        mods = proj.get_models()
        mod = dr.Model.get(projectId, modelId)
        feats = mod.get_features_used()

        # check if the post request has the file part
        if 'file' not in request.files:
            message = 'No file supplied'
            print("Message: ", message)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            message = 'empty filname'
            print("Message: ", message)
        nrows = 0
        ncols = 0
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pdata = pd.read_csv(filepath)
            nrows =  len(pdata)
            ncols = len(pdata.columns)

            total, optimised_lb, optimised_ub, f1c, f2c = opti.run_brute_force(proj, mod, pdata, colOne, colTwo)

            return render_template("runoptimization.html", project=proj, 
                                    models=mods, model=mod, total=total, features=feats, 
                                    optimised_lb=optimised_lb, optimised_ub=optimised_ub,
                                    colone=colOne, coltwo=colTwo,
                                    feat1_change=f1c, feat2_change=f2c)
    else:
        return "<h1>No Post Request - Invalid Request</h1><br/>"

# ###################################################################################
# About Page
@app.route('/about')
def about():
        return render_template("about.html")


# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
	app.run(port=5000, debug=True)


