# ##############################################################################################################
# SIMPLE OPTIMISATION
#
# NOTE: This approach makes some very strong asumptions about the model and problem:
#       Significantly, that the model captures the underlying probability relationship
#       with the independent variables and the dependent (of your problem) well enough that you
#       can potentially modify the distribution of the independent variables and 
#       still get meaningful results.
#
#       The goal of this project is to estimate the potential ROI outcome, presuming
#       that the causality holds and the model approximates this relationship.
# ##############################################################################################################

import datarobot as dr
import numpy as np
import pandas as pd

def get_midpoints_of_binned_intervals(pdata, colname):
    intervals = pdata[colname].value_counts(bins=30).index.tolist()
    mydist = pdata[colname].value_counts(bins=30).tolist()
    def interval_midpoint(x): 
        return (x.left + (x.right-x.left)/2)
    midpoints = [ interval_midpoint(i) for i in intervals]
    return midpoints, mydist


##################################################################################################################
# GET THE CANDIDATE LIST FOR SIMULATION
# INCLUDING THE EXISTING DISTRIBITION OVER THESE VALUES
# TODO : TEST SOME VARIATIONS ON THIS 
##################################################################################################################
def get_val_list_to_simulate(pdata, colname):
    colvals = pdata[colname].unique()
    if len(colvals)<31:
        myvals = pdata[colname].value_counts()
        valdist = np.array((myvals).tolist())
        return colvals, valdist
    else:
      myvals = pdata[colname].value_counts().index.tolist()[0:30]
      mydist = pdata[colname].value_counts().tolist()[0:30]
      if(pdata[colname].dtype == np.float64 or pdata[colname].dtype == np.int64):
          # CRUDE CHECK TO SEE IF THE DISTRIBUTION IS SKEWED TOWARDS A FEW VALUES
          if mydist[0] > (0.1*len(pdata)):
              return myvals, mydist
          else:
              myvals, mydist = get_midpoints_of_binned_intervals(pdata, colname)
              return myvals, mydist
      else: 
          return myvals, mydist


##################################################################################################################
# GET SIMULATED DATA
##################################################################################################################
def get_simulated_data(pdata, colone, colone_values, coltwo, coltwo_values):
    new_data = []
    for c1_value in colone_values:
        for c2_value in coltwo_values:
            temp_data = pdata.copy()
            temp_data[colone] = c1_value
            temp_data[coltwo] = c2_value
            new_data.append(temp_data)
    return pd.concat(new_data)

def sample_down(pdata):
    if len(pdata) < 1000 :
        return pdata
    rez = pdata.sample(1000).copy()
    return rez.reset_index()


##################################################################################################################
# RUN BRUTE FORCE
##################################################################################################################
def run_brute_force(project, model, df, colone, coltwo):
    # Separate the data into two sets for 
    # a potential calibration step.
    print("Total records %i" % len(df))
    pdata = sample_down(df)
    print("Sampled records %i" % len(pdata))
    positive_class = project.positive_class
    target = project.target
    midpoint = int(len(pdata)/2)
    print("Midpoint %i" % midpoint)
    pdata1 = pdata.loc[0:midpoint]
    pdata2 = pdata.loc[midpoint+1:]
    actuals1 = sum(pdata1[target] == positive_class)
    actuals2 = sum(pdata2[target] == positive_class)

    col1vals, col1dist = get_val_list_to_simulate(pdata, colone)
    col2vals, col2dist = get_val_list_to_simulate(pdata, coltwo)

    print("Col 1 contains %i Unique Values: " % len(col1vals) )
    print("Distribution", col1dist)
    print("Col 2 contains %i Unique Values: " % len(col2vals) )
    print("Distribution", col2dist)

    # Score the entire dataset as it is, then split (order is preserved)
    preds = get_scores(project, model, pdata)
    preds1 = preds.loc[0:midpoint] 
    preds2 = preds.loc[midpoint+1:]
    expected1 = sum(preds1['positive_probability'])
    expected2 = sum(preds2['positive_probability'])

    # #############################################################################################
    # Create an out-of-sample calibration factor for the expected outcome.
    # This analysis will depend to a large extent on how well we can estimate
    # a change in the expected number of a given outcome, which requires a
    # well calibrated model
    # #############################################################################################
    adjustment2 = ( actuals1 - expected1 ) / expected1
    adjustment1 = ( actuals2 - expected2 ) / expected2
    adjusted1 = expected1 + ( expected1 * adjustment1)
    adjusted2 = expected2 + ( expected2 * adjustment2)
   
    raw_exp_err1 = round( 100 *(expected1 - actuals1) / actuals1,1)
    raw_exp_err2 = round( 100 *(expected2 - actuals2) / actuals2,1)
    adj_exp_err1 = round( 100 *(adjusted1 - actuals1) / actuals1,1)
    adj_exp_err2 = round( 100 *(adjusted2 - actuals2) / actuals2,1)
    raw_lb = min(raw_exp_err1, raw_exp_err2)
    raw_ub = max(raw_exp_err1, raw_exp_err2)
    adj_lb = min(adj_exp_err1, adj_exp_err2)
    adj_ub = max(adj_exp_err1, adj_exp_err2)
 
    print("Subset 1. RAW ERROR  %f ADJUSTED: %f " % (raw_exp_err1, adj_exp_err1) )
    print("Subset 2. RAW ERROR  %f ADJUSTED: %f " % (raw_exp_err2, adj_exp_err2) )

    total = actuals1+actuals2
    raw_expected = round(expected1+expected2,1)
    adj_expected = round(adjusted1+adjusted2,1)

    print("Total. Actuals: %f Predicted: %f Adjusted: %f " % (total, raw_expected, adj_expected) )

    # ##############################################################################################
    # NOW GENERATE A DATASET THAT CONTAINS A LARGE NUMBER OF COMBINATIONS OF THE
    # TWO COLUMNS WE ARE ATTEMPTING TO OPTOMISE OVER. 
    # ##############################################################################################
    sim_data = get_simulated_data(pdata, colone, col1vals, coltwo, col2vals)

    records = len(pdata)
    print("Optimising %i Records" % (len(pdata)) )
    print("Scoring %i Permutations" % (len(sim_data)) )

    scored_data = get_scores(project, model, sim_data)
 
    keep_cols = [colone, coltwo, target]
    justcols = sim_data[keep_cols].copy()
    justcols = justcols.reset_index(drop=True)
    scored = scored_data.copy()
    scored.reset_index(drop=True)
    to_process = pd.concat([justcols, scored], axis=1)
 
    # NOW WE ITERATE OVER EACH OF THE ORIGINAL ROWS 
    # AND DETERMINE WHICH COMBINATION MAXIMISED THE PREDICTED TARGET
    # WE USE THE INDEXES TO RETRIEVE ALL COMBINATIONS BUILT OVER EACH
    # OF THE ORIGINAL RECORDS
    tempsum = 0
    onevals = []
    twovals = []
    for i in range(records):
        index_filter = records + 1
        temp_scores = to_process[(to_process.index == i) | ((to_process.index + 1) % index_filter == 0) ]
        actualval = preds['positive_probability'].loc[i]
        maxval, maxone, maxtwo = get_optimal_combination(temp_scores, colone, coltwo, actualval)
        tempsum = tempsum + maxval
        onevals.append(maxone)
        twovals.append(maxtwo)

    f1d = add_pseudo_counts(calculate_feature_distribution(col1vals, onevals))
    f2d = add_pseudo_counts(calculate_feature_distribution(col2vals, twovals))
 
    f1c = calculate_feature_distribution_change(col1dist, f1d)
    f2c = calculate_feature_distribution_change(col2dist, f2d)

    # ##############################################################################################
    # RATHER THAN RETURN THIS DIRECTLY WE USE THE OBSERVED ERROR IN PREDICTING THE TARGET FROM OUR 
    # TWO OUT_OF_SAMPLE SETS ABOVE, TO CREATE A LOWER AND UPPER ADJUSTED ESTIMATE OF THE OPTIMISED
    # TARGET BASED ON THE CHANGED INPUTS.
    # NOTE: THIS IS STILL VERY CRUDE. IT WOULD BE BETTER IF THE ERROR ESTIMATE WAS SENSITIVE TO THE
    #       INPUT FEATURES, SO WE KNEW IF WE HAD OPTIMISED TO A POORLY REPRESENTED REGION IN THE 
    #       FEATURE SPACE.
    # TODO: ADD IN A MEASURES OF THE CHANGE IN THE INPUT DISTRIBUTION OF THE OPTIMSED FEATURES
    # ##############################################################################################

    adj_exp_1 = tempsum + ( tempsum * adjustment1)
    adj_exp_2 = tempsum + ( tempsum * adjustment2)
    lb = min(adj_exp_1, adj_exp_2)
    ub = max(adj_exp_1, adj_exp_2)
    return total, lb, ub, f1c, f2c

##################################################################################################################
# CALCULATE THE FEATURE DISTRIBUTION CHANGE - KULLBACK Leibler
##################################################################################################################
def add_pseudo_counts(counts):
    return list(map(lambda x: x+1, counts))


##################################################################################################################
# CALCULATE THE FEATURE DISTRIBUTION CHANGE - KULLBACK Leibler
##################################################################################################################
def calculate_feature_distribution(bins, values):
    temp = np.zeros(len(bins))
    mybins = bins.tolist()
    myvals = values #.tolist()
    print(mybins)
    for i in range(len(myvals)):
        #print("testing for value:", myvals[i])
        temp[mybins.index(myvals[i])] += 1
    return temp

##################################################################################################################
# CALCULATE THE FEATURE DISTRIBUTION CHANGE - TODO
##################################################################################################################
def calculate_feature_distribution_change(original, optimised):
    # INPUT MAY BE COUNT ARRAYS - SO NORMALISE AS PROBABILITY DISTRIBUTIONS
    act = np.array(original)/sum(original)
    print("Original Distribution:", act)
    mod = np.array(optimised)/sum(optimised)
    print("Optimised Distribution:", mod)
    # THEN RETURN KL DIVERGENCE
    return (mod * np.log(mod/act)).sum()


##################################################################################################################
# FIND THE OPTIMAL VALUES FOR THIS ROW.
##################################################################################################################
def get_optimal_combination(sim_rows, colone, coltwo, actualval):
    max_row = sim_rows[sim_rows['positive_probability']==max(sim_rows['positive_probability'])]
    maxval = float(max_row["positive_probability"]) 
    maxone = max_row[colone].values[0]
    maxtwo = max_row[coltwo].values[0]
    return maxval, maxone, maxtwo


##################################################################################################################
# DEPRECATED - THIS WAS A VERY BAD WAY TO DO THIS
##################################################################################################################
def optimise_predicted_output(project, model, colone, col1vals, coltwo, col2vals, temprow, actualval):
    maxone = temprow[colone]
    maxtwo = temprow[coltwo]
    maxval = actualval
    for v1 in col1vals:
        for v2 in col2vals:
            temprow[colone] = v1 
            temprow[coltwo] = v2
            rez = get_scores(project, model, temprow)
            score = rez['positive_probability'].loc[0]
            if score > maxval:
                maxval = score
                maxone = v1
                maxtwo = v2
    return maxval, maxone, maxtwo


##################################################################################################################
# GET THE SCORES - 
##################################################################################################################
def get_scores(project, model, pdata):
    dataset = project.upload_dataset(pdata)
    pred_job = model.request_predictions(dataset.id)
    preds = dr.models.predict_job.wait_for_async_predictions(project.id, predict_job_id=pred_job.id, max_wait=600)
    return preds


