DataRobot ROI Estimation Tool
========================================================

The goal of this project is to demonstrate several methods of
estimating the ROI of a model prior to running a live test.

### Assumptions

This project assumes you have a valid DataRobot account and that you
have set up your account credentials in the drconfig.yaml file so that
you can use the API.
 
We assume that you have python installed with the DataRobot package.

We assume that you have an existing DataRobot project with a set of models
for which you want to estimate the protenital ROI.

### How to use it

Run the application with
```
python app.py
```

Then point your browser at the URL provided when the flask container is launched.

The web application will guide you through the process of choosing a project
and then the method of estimating ROI you want to use.


