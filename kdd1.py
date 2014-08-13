import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split

filepath = '/Users/rowanvasquez/Documents/Data Science/Kaggle/KDD Cup'
outcomes_file = filepath + '/outcomes.csv'
projects_file = filepath + '/projects.csv'
resources_file = filepath + '/resources.csv'
donations_file = filepath + '/donations.csv'
samplesub_file = filepath + '/sampleSubmission.csv'



out = pd.read_csv(outcomes_file)
proj = pd.read_csv(projects_file)
res = pd.read_csv(resources_file)
don = pd.read_csv(donations_file)
sample = pd.read_csv(samplesub_file)

out.head()
proj.head()
res.head()
don.head()
sample.head()



#Look at the is_exciting var distribution
out.is_exciting.value_counts()



out_simple = out[['projectid', 'is_exciting']] #just the id and exciting columns


proj1 = pd.concat([out_simple, proj]) 
proj = proj1


#splitting into the final testing and training set
proj_kaggle_train = proj[proj['date_posted'] >= '2014-01-01']
proj_kaggle_test = proj[proj['date_posted'] < '2014-01-01']
len(proj_kaggle_test)
len(proj_kaggle_train)



#split the final testing set into a training and tesitng set

proj_train, proj_test = train_test_split(proj_kaggle_train, train_size = 0.7, test_size = 0.3, random_state = 352)
proj_train = pd.DataFrame(proj_train)
proj_test = pd.DataFrame(proj_test)
proj_train.columns = proj_test.columns = proj_kaggle_train.columns

len(proj_test)
len(proj_train)

#Baseline model:
#No project is exciting

#For testing data set, that gives us
sum_values = (proj_test.is_exciting.value_counts()[0] + proj_test.is_exciting.value_counts()[1])

baseline_accur = proj_test.is_exciting.value_counts()[1] / sum_values
#baseline_accur = 174695/185798
#94.702%
