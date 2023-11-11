import pandas as pd
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler


def preprocessing(df, upsampling_ratio=None, downsampling_ratio=None, test_size=0.3):
    '''
        Preprocessing pipeline which performs a one-hot-encoding for all categorical features and returns a train-test-split
    '''
    processed = df.copy()

    #Drop id column
    processed.drop('id', axis=1, inplace=True)

    #Mark Policy_Sales_Channel and Region_Code as categorical columns such that get_dummies performs 1-hot-encoding on them
    processed['Policy_Sales_Channel'] = processed['Policy_Sales_Channel'].astype('int32').astype('category')
    processed['Region_Code'] = processed['Region_Code'].astype('int32').astype('category')

    #Create new features
    #processed['Premium_Age_Ratio'] = processed['Annual_Premium']/processed['Age']
    #processed['Premium_Vintage_Ratio'] = processed['Annual_Premium']/processed['Vintage']
    processed['Age_sq'] = processed['Age']*processed['Age']
    processed['Annual_Premium_sq'] = processed['Annual_Premium']*processed['Annual_Premium']
    processed['Vintage_sq'] = processed['Vintage']*processed['Vintage']


    #Reorder columns such that response/label is the last column
    processed = processed[[c for c in processed if c not in ['Response']]+ ['Response']]

    #Apply StandardScaler to numerical variables for the neural network downstream
    #processed[['Age','Annual_Premium','Vintage']] = StandardScaler().fit_transform(processed[['Age','Annual_Premium','Vintage']])
    processed[['Age','Annual_Premium','Vintage','Age_sq','Annual_Premium_sq','Vintage_sq']] = StandardScaler().fit_transform(processed[['Age','Annual_Premium','Vintage','Age_sq','Annual_Premium_sq','Vintage_sq']])

    #Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(processed.iloc[:,:-1], processed.iloc[:,-1], test_size=test_size, random_state=42)

    #random downsampling
    if downsampling_ratio:
        X_train, y_train = RandomUnderSampler(sampling_strategy=downsampling_ratio, random_state=7).fit_resample(X_train,y_train)
    
    #upsampling with smote
    if upsampling_ratio:
        smote = SMOTENC(categorical_features=[0,2,3,4,5,7], sampling_strategy=upsampling_ratio, n_jobs=-1)
        #smote = SMOTENC(categorical_features=[0,2,3,4,5], sampling_strategy=upsampling_ratio, n_jobs=-1, random_state=5)
        X_train, y_train = smote.fit_resample(X_train,y_train)

    #OneHotEncoding    
    X_train = pd.get_dummies(X_train,drop_first=True)  
    X_test = pd.get_dummies(X_test,drop_first=True)  
    X_train.rename(columns={'Vehicle_Age_< 1 Year': 'Vehicle_Age_smallerthan_1 Year', 'Vehicle_Age_< 2 Years': 'Vehicle_Age_greaterthan_2 Years'}, inplace=True)
    X_test.rename(columns={'Vehicle_Age_< 1 Year': 'Vehicle_Age_smallerthan_1 Year', 'Vehicle_Age_< 2 Years': 'Vehicle_Age_greaterthan_2 Years'}, inplace=True)

    
    return X_train, X_test, y_train, y_test 


def eval_clf_perf(clf, X_test, y_test, threshold=0.5):
    '''
        Calculates important metrics to evaluate the performance of a classifier
    '''
    y_probs = clf.predict_proba(X_test)[:,1]
    y_pred = np.zeros(y_probs.shape[0])
    for i in range(y_probs.shape[0]):
        if y_probs[i] >= threshold:
            y_pred[i] = 1
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {}
    metrics['accuracy'] = (tn + tp)/(tn+tp+fn+fp)
    metrics['sensitivity'] = (tp)/(tp+fn)
    metrics['precision'] = tp/(tp+fp)
    metrics['fscore'] = 2*metrics['precision']*metrics['sensitivity']/(metrics['precision']+metrics['sensitivity'])
    metrics['specificity'] = tn/(tn+fp)
    metrics['negative_precision'] = tn/(tn+fn)
    for key in metrics:
        print(key + ': ' + str(metrics[key]))
    return metrics


## Displays the calibration curve which helps to evaluate how close the predicted class probabilities correspond to actual probabilites
def eval_clf_calib(clf, X_test, y_test, clf_name=''):
    probs = clf.predict_proba(X_test)[:,1] #probabilities of class 1
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='quantile')
    disp = CalibrationDisplay(prob_true, prob_pred, probs, estimator_name=clf_name)
    #disp.ax_.set_title('Calibration Curve')
    disp.plot(color='#9659F4')
    plt.ylabel('Mean empirical probability')
    plt.savefig('../CalibrationCurve2', dpi=500)
    #plt.title('Calibration Curve')
    #plt.show()


## Calculates the ROC curve and AUC score
def eval_clf_roc(clf, X_test, y_test, clf_name=''):
    fpr, tpr, _ = roc_curve(y_test.values, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=clf_name)
    display.plot()
    plt.show()


## Plot histogram of the predicted class probabilites
def class_probs_hist(clf, X_test, clf_name=''):
    probs = clf.predict_proba(X_test)[:,1]
    ax = sns.histplot(100*probs, stat='percent', color='#9659F4', bins=15)
    ax.set(xlabel='Purchase Propensity [%]', title=clf_name, ylabel='Percentage of Occurance [%]')
    ax.set_yticks(list(range(0,51,5)))
    ax.set_xticks(list(range(0,61,10)))
    plt.savefig('../PurchaseProbs15.png', dpi=500)
    plt.show()


## Bundle evaluation functions
def eval_clf(clf, X_test, y_test, clf_name='', threshold=0.5):
    metrics = eval_clf_perf(clf, X_test, y_test, threshold)
    eval_clf_calib(clf, X_test, y_test)
    eval_clf_roc(clf, X_test, y_test)
    class_probs_hist(clf, X_test, clf_name)
    return metrics