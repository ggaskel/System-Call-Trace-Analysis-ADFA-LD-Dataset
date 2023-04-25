#Import required packages
import os
import math
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression

#Used to format name based on list of n-gram terms
def format_name(myList):
    if len(myList)==1: return str(myList[0])+'-gram'
    elif len(myList)==2: return str(myList[0])+' & '+str(myList[1])+'-gram'
    else: return str(myList[0])+', '+str(myList[1])+' & '+str(myList[2])+'-gram'

#Function to calculate precision (note: may be undefined if alg doesn't classify any traces as attacks)
def precision(actual_y,pred_y):
    tp_fp = sum(pred_y)
    if tp_fp == 0:
        return float('nan')
    tp = 0
    for i in range(len(actual_y)):
        if actual_y[i] ==1 and pred_y[i]==1:
            tp+=1
    return(tp/tp_fp)

#Function to calculate recall
def recall(actual_y,pred_y):
    tp_fn = sum(actual_y)
    tp = 0
    for i in range(len(actual_y)):
        if actual_y[i] == 1 and pred_y[i] == 1:
            tp+=1
    return(tp/tp_fn)

def fprate(actual_y,pred_y):
    fp_tn = len(actual_y) - sum(actual_y)
    fp = 0
    for i in range(len(actual_y)):
        if pred_y[i] == 1 and actual_y[i] ==0:
            fp+=1
    return(fp/fp_tn)

#Function to calculate F-measure (note: will be undefined if precision is undefined, or precision+recall=0)
def fmeasure(calc_precision,calc_recall):
    if (math.isnan(calc_precision) == False) and (math.isnan(calc_recall) == False):
        if calc_precision + calc_recall == 0:
            return 'nan'
        else:
            return((2*calc_precision*calc_recall)/(calc_precision+calc_recall))
    else:
        return float('nan')

#Function to manually calculate accuracy
def accuracy(actual_y,pred_y):
    correct = 0
    for i in range(len(actual_y)):
        if actual_y[i]==pred_y[i]:
            correct+=1
    return(correct/len(actual_y))

#Function to calculate average of list values (used to compute final CV model results)
def average(myList):
    if len(myList) == 0:
        return float('nan')
    sum = 0
    num_elem = 0
    for elem in myList:
        if math.isnan(float(elem)) == False:
            sum += elem
            num_elem += 1
    if num_elem ==0:
        return float('nan')
    return sum/num_elem

#SCENARIOS TO TEST
scenarios = [([1],175),([2],3792),([3],24818),([4],79962),([5],163263),([1,2],3967),([1,3],24993),([1,4],80137),([1,5],163438),([2,3],28610),([2,4],83754),([2,5],167055),([3,4],104780),([3,5],188081),([4,5],243225),([1,2,3],28785),([1,2,4],83929),([1,2,5],167230),([1,3,4],104955),([1,3,5],188256),([1,4,5],243400),([2,3,4],108572),([2,3,5],191873),([2,4,5],247017),([3,4,5],268043)]

roc = False
for scen in scenarios:
    #Read in the feature vectors .csv file and separate features/label data
    #Parameters to call different datasets and adjust % principal components retained
    n = scen[0]
    num_features = scen[1]
    calc_type = 'tfidf'      #options: tfidf or freq or bool
    per_pc = 1               #options: 1,2,3,4,5
    s =''
    for elem in n:
        s+=str(elem)
    filename = 'test_data'+s+'_'+calc_type+'.csv'

    os.chdir("C:/Users/grego/PycharmProjects/thesis")
    labels = np.loadtxt(filename, skiprows = 1,delimiter=',',dtype = int,usecols=(0))
    features = np.loadtxt(filename, skiprows = 1,delimiter=',',usecols=range(1,num_features+1))

    print(format_name(n))

    svd = TruncatedSVD(n_components=int(round(num_features * per_pc * .01, 0)), n_iter=1, random_state=42)
    svd.fit(features)
    features_red = svd.transform(features)
    print("Explained Var:",svd.explained_variance_ratio_.sum())
    print()
    #Creates lists to store 5 metrics for each method for cross validation
    lda_accuracy_list = []
    lda_precision_list = []
    lda_recall_list = []
    lda_fmeasure_list = []
    lda_fprate_list =[]
    rfc_accuracy_list = []
    rfc_precision_list = []
    rfc_recall_list = []
    rfc_fmeasure_list = []
    rfc_fprate_list =[]
    lr_accuracy_list = []
    lr_precision_list = []
    lr_recall_list = []
    lr_fmeasure_list = []
    lr_fprate_list =[]

    if roc == True:
        #Plotting ROC Curves
        #For LDA
        tprsL = []
        aucsL = []
        mean_fprL = np.linspace(0, 1, 100)
        figL, axL = plt.subplots()
        #For RFC
        tprsR = []
        aucsR = []
        mean_fprR = np.linspace(0, 1, 100)
        figR, axR= plt.subplots()
        #For Log Reg
        tprsLR = []
        aucsLR = []
        mean_fprLR= np.linspace(0, 1, 100)
        figLR, axLR = plt.subplots()

    #Implements stratified k-fold cross validation and obtains metrics for each model (k times)
    x = features_red
    y = labels
    #Implementation of k-fold CV
    kf = StratifiedKFold(n_splits=10,shuffle=True)
    fold_num = 0
    for train_index, test_index in kf.split(x,y):
        fold_num+=1
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Obtains model metrics and stores them in lists
        #LDA Model
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train,y_train)
        obt_accuracyL = lda.score(x_test,y_test)
        predsL = lda.predict(x_test)
        lda_accuracy_list.append(obt_accuracyL)
        obt_precisionL = precision(y_test,predsL)
        lda_precision_list.append(obt_precisionL)
        obt_recallL = recall(y_test,predsL)
        lda_recall_list.append(obt_recallL)
        lda_fmeasure_list.append(fmeasure(obt_precisionL,obt_recallL))
        lda_fprate_list.append(fprate(y_test,predsL))


        #Random Forest Classification Model
        rfc = RandomForestClassifier()
        rfc.fit(x_train,y_train)
        obt_accuracyF = rfc.score(x_test,y_test)
        predsF = rfc.predict(x_test)
        rfc_accuracy_list.append(obt_accuracyF)
        obt_precisionF = precision(y_test,predsF)
        rfc_precision_list.append(obt_precisionF)
        obt_recallF = recall(y_test,predsF)
        rfc_recall_list.append(obt_recallF)
        rfc_fmeasure_list.append(fmeasure(obt_precisionF,obt_recallF))
        rfc_fprate_list.append(fprate(y_test,predsF))


        #Logistic Regression Model
        lr = LogisticRegression(max_iter = 10000)
        lr.fit(x_train, y_train)
        obt_accuracyLR = lr.score(x_test, y_test)
        predsLR = lr.predict(x_test)
        lr_accuracy_list.append(obt_accuracyLR)
        obt_precisionLR = precision(y_test, predsLR)
        lr_precision_list.append(obt_precisionLR)
        obt_recallLR = recall(y_test, predsLR)
        lr_recall_list.append(obt_recallLR)
        lr_fmeasure_list.append(fmeasure(obt_precisionLR, obt_recallLR))
        lr_fprate_list.append(fprate(y_test, predsLR))
        if roc ==True:
            #For ROC Curves (LDA)
            vizL = RocCurveDisplay.from_estimator(lda,x_test,y_test,name="ROC Fold {}".format(fold_num),alpha=0.3,lw=1,ax=axL)
            interp_tprL = np.interp(mean_fprL, vizL.fpr, vizL.tpr)
            interp_tprL[0] = 0.0
            tprsL.append(interp_tprL)
            aucsL.append(vizL.roc_auc)

            #For ROC Curves (RFC)
            vizR = RocCurveDisplay.from_estimator(rfc, x_test, y_test, name="ROC Fold {}".format(fold_num), alpha=0.3, lw=1,ax=axR)
            interp_tprR = np.interp(mean_fprR, vizR.fpr, vizR.tpr)
            interp_tprR[0] = 0.0
            tprsR.append(interp_tprR)
            aucsR.append(vizR.roc_auc)

            # For ROC Curves (LR)
            vizLR = RocCurveDisplay.from_estimator(lr, x_test, y_test, name="ROC Fold {}".format(fold_num), alpha=0.3, lw=1, ax=axLR)
            interp_tprLR = np.interp(mean_fprLR, vizLR.fpr, vizLR.tpr)
            interp_tprLR[0] = 0.0
            tprsLR.append(interp_tprLR)
            aucsLR.append(vizLR.roc_auc)

    #Summary of Results from Cross Validation (averages of metrics)
    print('LDA RESULTS:')
    print(average(lda_accuracy_list)) #Accuracy
    print(average(lda_precision_list)) #Precision
    print(average(lda_recall_list)) #Recall
    print(average(lda_fmeasure_list)) #F-Measure
    print(average(lda_fprate_list)) #FP Rate
    print()
    print('RANDOM FOREST CLASSIFICATION RESULTS:')
    print(average(rfc_accuracy_list))
    print(average(rfc_precision_list))
    print(average(rfc_recall_list))
    print(average(rfc_fmeasure_list))
    print(average(rfc_fprate_list))
    print()
    print('LOGISTIC REGRESSION RESULTS:')
    print(average(lr_accuracy_list))
    print(average(lr_precision_list))
    print(average(lr_recall_list))
    print(average(lr_fmeasure_list))
    print(average(lr_fprate_list))
    print()
    if roc == True:
        #Generating ROC Curves for Classifiers
        #LDA ROC Plot
        mean_tprL = np.mean(tprsL, axis=0)
        mean_tprL[-1] = 1.0
        mean_aucL = auc(mean_fprL, mean_tprL)
        axL.plot(mean_fprL,mean_tprL,color="b",label=r"Mean ROC (AUC = %0.2f)" % mean_aucL,lw=2,alpha=0.8)
        LDA_title = "LDA ROC Curves"
        axL.set(xlim=[-0.05, 1.05],ylim=[-0.05, 1.05],title=LDA_title)
        axL.legend(loc="lower right")
        #RFC ROC Plot
        mean_tprR = np.mean(tprsR, axis=0)
        mean_tprR[-1] = 1.0
        mean_aucR = auc(mean_fprR, mean_tprR)
        axR.plot(mean_fprR,mean_tprR,color="b",label=r"Mean ROC (AUC = %0.2f)" % mean_aucR,lw=2,alpha=0.8)
        RFC_title = "Random Forest ROC Curves"
        axR.set(xlim=[-0.05, 1.05],ylim=[-0.05, 1.05],title=RFC_title)
        axR.legend(loc="lower right")
        # Log Reg ROC Plot
        mean_tprLR = np.mean(tprsLR, axis=0)
        mean_tprLR[-1] = 1.0
        mean_aucLR = auc(mean_fprLR, mean_tprLR)
        axLR.plot(mean_fprLR, mean_tprLR, color="b", label=r"Mean ROC (AUC = %0.2f)" % mean_aucLR, lw=2, alpha=0.8)
        LR_title = "Logistic Regression ROC Curves"
        axLR.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=LR_title)
        axLR.legend(loc="lower right")

        #Saves figs to appropriate folder
        dir = "C:/Users/grego/PycharmProjects/thesis"
        os.chdir(dir)
        #Random Forest figure
        figR_filename = 'bool_roc_rf'+'.png'
        figR.savefig(figR_filename, bbox_inches='tight')
        #LDA figure
        figL_filename = 'bool_roc_lda'+'.png'
        figL.savefig(figL_filename, bbox_inches='tight')
        #Log Reg figure
        figLR_filename = 'bool_roc_lr'+'.png'
        figLR.savefig(figLR_filename, bbox_inches='tight')

        print('Figures saved.')
        print()
        plt.close('all')