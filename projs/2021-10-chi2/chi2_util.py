import scipy.stats as scs
import pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# This function ranks the feature, according to their corrleation to the target feature.
# The features must be categorical. The ranking is done with scipy.stats.chi2_contingency()
# returns: dataframe of features, sorted descending by correlation.
# Each feature comes with chi2 statistics, p-value, critical value, rank, and reversed rank.
# There are two points (marked in the code) where the logic is simplified and could be improved.
#
# parameters:
# df = the source data
# features = list of columns (features) of the source data that should be ranked
# target = the name of
# verbose = print what you are doing
# deep = if True, then once the contigency table for a given feature has been built, 
# the rows with too few values will be removed, before calculating chi2 statistics.
# As an experimental feature, one can turn this condition to False, 
# which will cause calculating the chi2 on all rows of the rc table.


def chi2_score(df, features, target, alpha, verbose = False, deep = True):
    
    # create the 'id' column to contain unique identifier. 
    if (not df.index.is_unique):
        df = df.reset_index(drop = True)
    df['id'] = df.index

    scoring = pd.DataFrame()
    prob = 1 - alpha
    if (verbose): 
        print('Contingency tables:\n ')
    for i in features:
        
        # create the contingency table (rc table)
        rc = df.pivot_table(values = 'id', columns = target, index = i, aggfunc = len).fillna(0)
               
        # remove the rc table rows with too few values, because the chi2 results can be unreliable otherwise
        if (deep):
                # below a quick hack, rather than prod: we are cutting off all rows of low observed values.
                # not exactly correct. The proper condition is:
                # No more than 20% of the expected (sic!) counts are less than 5 and all individual expected counts are 1 or greater
                
                # keep the rows with sum of frequencies bigger than limit
                totals = rc.sum(axis = 1)
                limit = 10 #arbitrary
                high_freq = rc[totals>= limit]
                # aggregate the remaining rows into one row
                #  representing "all rows with low frequencies"
                # and add that row to rc
                low_freq = rc[totals <limit].sum(axis = 0)
                # concatenate high and low frequency rows
                rc = high_freq.append(low_freq, ignore_index = True)
        if (verbose):
            print(rc)
            print()
                
        # calculate chi2 statistics based on rc table
        chi2s, p, dof, expected = scs.chi2_contingency(rc)
        
        # keep the results in the scoring chart
        row = pd.Series({'chi2': chi2s, 
                         'dof': dof, 
                         'critical': scs.chi2.ppf(prob, dof), 
                         'p': p}, 
                        name = i)
        scoring = scoring.append(row)
      
    # The rows below calculate the rank, and sort the features accordingly
    
    # how do we rank the features? First of all, by the p-value (lowest p-value wins).
    # secondly if the p value is the same for two features, then the winning feature
    # is the one whose chi2 score exceeds the critical value by higher factor
    # (Note: I am not sure if this is the correct way)

    # how many times does chi2 statistics exceed the critical value
    scoring['rank'] = scoring['chi2'] / scoring['critical'] 
    # we need the reverse rank just for the purpose of ascending sorting
    scoring['reverse_rank'] = 1 / scoring['rank']
    # features with lower p-value are more meaningful
    # if two features have same p-value, then feature with rank order by p value
    scoring = scoring.sort_values(by = ['p', 'reverse_rank'], ascending = True)
    return scoring    


# A utility returning an accuracy of a given classifier 
# on a given set of X, y (features, label)
# Note: the set should be balanced, otherwise the accuracy may be a wrong measure

def quick_accuracy(X, y, classifier):
    seed = 42
    test_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = classifier
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
    
# A utility, returning an accuracy of a given classifier
# on a given set of X, y (features, label), separately for each feature.
# The classifier is run several times, each time given only one feature as input.
# The input set should be balanced, otherwise the accuracy may be a wrong measure
def accuracy_by_feature(X, y, classifier):
    #print("feature         : accuracy")
    df = pd.DataFrame()
    for c in X.columns:
        a = quick_accuracy(X[[c]], y, classifier)
        #print(f"{c:<10}\t: {(a * 100):.2f}")#.format(c, a * 100.0))
        df = df.append({'feature': c, 'accuracy': a}, ignore_index = True)
    return df.sort_values(by = 'accuracy', ascending = False)
        
