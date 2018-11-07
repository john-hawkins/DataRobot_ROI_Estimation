import datarobot as dr

def evalBinaryClassModels(project, models, num_models, tp, fp, tn, fn, cases, baserate):
   results = []
   index = 0
   for mod in models:
      result = {}
      result['index'] = index
      result['model_type'] = mod.model_type
      result['sample_pct'] = mod.sample_pct
      result['features'] = mod.featurelist_name
      result['metric'] = mod.metrics[ project.metric ]['validation']
      if index < num_models: 
          opti = estimateOptimalThreshold(mod, tp, fp, tn, fn, cases, baserate)
          result['threshold'] = round(opti['threshold'],3)
          result['roi'] = round(opti['roi'], 0)
      else:
          result['threshold'] = "?"
          result['roi'] = "?"
      results.append(result)
      index = index+1
   return results


def estimateOptimalThreshold(mod, tp, fp, tn, fn, cases, baserate):
   rocCurve = mod.get_roc_curve('validation')
   points = rocCurve.roc_points
   pos = cases * baserate
   neg = cases - pos
   thresh = 1.0
   roi = -999999
   for point in points:
      temp = (neg * point['true_negative_rate'] * tn) +\
      (neg * point['false_positive_rate'] * fp ) +\
      (pos * point['true_positive_rate'] * tp ) +\
      (pos * (1-point['true_positive_rate']) * fn)
      if temp>roi:
         roi = temp
         thresh = point['threshold']
   return {'threshold': thresh, 'roi': roi}


def convertIntervention(cases, baserate, cost, payoff, payback, succrate, backfire):
   tp = payoff*succrate - cost
   fp = payback*backfire - cost
   tn = 0 
   fn = 0
   print("True Positive Payoff: ", tp )
   print("False Positive Payoff: ", fp )
   return tp, fp, tn, fn

 
