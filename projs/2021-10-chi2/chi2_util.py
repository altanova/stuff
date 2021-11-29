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
# deep = if True (detault), then the RC table calculated with proper chi2 conditions, that 
# is: 80% cells must have expected count of 5 or more. How this works: the rows that have cells
# with expected count <5 are aggregated into one row named OTHER, before calculating the chi2 statistics.
# That artificial row will in most cases have high expected count, because it results from aggregation.
# So strictly speaking, we are even stricter than the quoted condition and implemented the following:
# All cells have expected count 5 or more, except (in rare situations) some cells in one row named OTHER.
# As an experimental feature, one can turn this variable to False, 
# which will cause calculating the chi2 on all rows of the rc table, regardless their count.
# According to literature this can return untrusted results, so True is recommended.

def chi2_score(df, features, target, alpha, verbose = False, deep = True):
    
    # create the 'id' column to contain unique identifier. 
    if (not df.index.is_unique):
        df = df.reset_index(drop = True)
    df['id'] = df.index

    scoring = pd.DataFrame()
    prob = 1 - alpha
    for i in features:
        
        # create the contingency table (rc table)
        rc = df.pivot_table(values = 'id', columns = target, index = i, aggfunc = len).fillna(0)
        if (verbose):
            print(f'feature {i}:\nContingency table')
            print(rc)
            print()               
        # remove the rc table rows with too few values, because the chi2 results can be unreliable otherwise
        if (deep):
                # calculation of the expected values
                # this is needed to calculate the condition for chi2:
                # each cell should have expected count >5
                # note some authores say 80% of the cells. We will strictly enforce all
                row_totals = rc.sum(axis = 1).values[:, np.newaxis]
                # category crequencies
                cfreq = (rc.sum(axis = 0) / rc.values.sum()).values[np.newaxis, :]
                
                if (verbose):
                    print('The expected distribution of categories:')
                    print(cfreq)
                    print()
                # the dataframe of expected values
                expected = pd.DataFrame(np.dot(row_totals, cfreq),  
                             index = rc.index, 
                             columns = rc.columns)
                limit = 5
                # rows that meet chi2 condition: expected >5 in each cell
                high_freq = rc[(expected >= limit).any(axis = 1)]
                # rows that do not meet that condition
                low_freq = rc[(expected < limit).any(axis = 1)]
                if (len(low_freq) > 0):
                    # concatenate high and low frequency rows
                    # Note there is possibility that only 1 row will be here, and so
                    # this row will have expected count <5.
                    # this is still ok, as most authors say that not all rows, but
                    # 80% of rows should have the expected count >=5
                    other = pd.Series(low_freq.sum(axis = 0), name = 'OTHER')
                    rc = high_freq.append(other)
                else:
                    rc = high_freq

        if (verbose):
            print('Contingency table after removing low frequency rows')
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
    # drop the temporary column id
    df.drop('id', axis=1, inplace=True)
    
    # The code below calculates the rank, and sorts the features accordingly
    
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
        
