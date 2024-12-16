###########IMPORT LIBRARIES###############
import pandas as pd
import numpy as np
import pyodbc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import pickle

from ModelPreprocessing import func_import_data

###IMPORT DATA###
print("IMPORT DATA")
dfX, dfY, dfGroups = func_import_data(30)

###SPLIT & SAMPLE_WEIGHT
print("SPLIT & SAMPLE_WEIGHT")
X=np.asarray(dfX.drop("SteelFamCluster",axis=1))
SFC_array=np.asarray(dfX.SteelFamCluster).reshape(-1,1)
y=np.asarray(dfY)
groups=np.asarray(dfGroups.group)
scaler=StandardScaler().fit(X)#Dont scale categorical encoded vars(SteelFamCluster)
X=scaler.transform(X)
X=np.concatenate((X,SFC_array),axis=1)
#X=np.concatenate((X,groups.reshape(-1,1)),axis=1)#[:,-1]

#Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42,stratify=groups)#Keeps % of groups in the splits, not group splitting
#[x for x in Xtrain[:,-1].tolist() if x in Xtest[:,-1].tolist()]
gss=GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    train_idx=train_index
    test_idx=test_index
#Check groups splitted: OK
Xtrain=X[train_idx]
Xtest=X[test_idx]
ytrain=y[train_idx]
ytest=y[test_idx]

sample_weight=compute_sample_weight(class_weight="balanced", y=X[:,-1])
train_sample_weight=compute_sample_weight(class_weight="balanced", y=Xtrain[:,-1])
test_sample_weight=compute_sample_weight(class_weight="balanced", y=Xtest[:,-1])


###############################################TRAIN MODEL###################################################################

##############################TREE REGRESSOR#################################
print("Tree regressor")
#:Grid 24min-288+3264 models
#If criterion (MSE or Poisson) predicted value is mean of terminal nodes. If criterion=MAE predicted value is median.
#Prevent overffitting with min_samples_leaf+max_depth or ccp_alpha

#U tell the grid which hyperparam it refers too by setting the name of the transformer (poly) with '__' and the name of the parameter (degree).
grid1={"splitter":["best","random"],'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"max_features":["sqrt","log2",None],"ccp_alpha":np.logspace(-5,-1,12)}#,"min_samples_leaf":np.arange(1,9),"max_leaf_nodes":
grid2={"splitter":["best","random"],'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"max_features":["sqrt","log2",None],"max_depth":np.arange(3,20),"min_samples_leaf":np.arange(4,12)}#,"min_samples_leaf":np.arange(1,9),"max_leaf_nodes":
tree=DecisionTreeRegressor(random_state=42)
cv=GroupShuffleSplit(n_splits=5, train_size=.8, random_state=42)
#USING SNAP
# snapgrid1={"splitter":["best"],'criterion':['mse'],"max_features":["sqrt","log2",None],"ccp_alpha":np.logspace(-5,-1,12)}#,"min_samples_leaf":np.arange(1,9),"max_leaf_nodes":
# sanpgrid2={"splitter":["best"],'criterion':['mse'],"max_features":["sqrt","log2",None],"max_depth":np.arange(3,20),"min_samples_leaf":np.arange(4,12)}#,"min_samples_leaf":np.arange(1,9),"max_leaf_nodes":
# snaptree=SnapDecisionTreeRegressor(random_state=42,n_jobs=-1)

#Train and save Tree models with alpha
GridTree1=GridSearchCV(estimator=tree,param_grid=grid1,scoring=['r2',"neg_root_mean_squared_error"],return_train_score=True,refit='r2',cv=cv,n_jobs=-1,verbose=3)#error_score='raise',
GridTree1.fit(Xtrain,ytrain[:,0].reshape(1,-1)[0],groups=groups[train_idx],sample_weight=train_sample_weight)
# Serialization
with open("GridTree1AlphaModels.pickle", "wb") as outfile:
    pickle.dump(GridTree1, outfile)

#Train and save Tree models with min_samples & max_depth
GridTree2=GridSearchCV(estimator=tree,param_grid=grid2,scoring=['r2',"neg_root_mean_squared_error"],return_train_score=True,refit='r2',cv=cv,n_jobs=-1,verbose=3)#error_score='raise',
GridTree2.fit(Xtrain,ytrain[:,0].reshape(1,-1)[0],groups=groups[train_idx],sample_weight=train_sample_weight)
# Serialization
with open("GridTree2DepthModels.pickle", "wb") as outfile:
    pickle.dump(GridTree2, outfile)
print("Written object", GridTree2)

#Save best model of each
data1=pd.DataFrame(GridTree1.cv_results_).loc[pd.DataFrame(GridTree1.cv_results_).rank_test_r2==1,:]
data2=pd.DataFrame(GridTree2.cv_results_).loc[pd.DataFrame(GridTree2.cv_results_).rank_test_r2==1,:]
if data1.ndim>1:
    data1=data1.iloc[0,:]
if data2.ndim>1:
    data2=data2.iloc[0,:]
dfTrees=pd.concat([data1,data2],axis=1).T
dfTrees.index=['alpha','depth_leaf']
#dfTrees = pd.DataFrame(columns=data2, index = 'depth_leaf')

#TAKE BEST TREE HYPERPARAM
dfTree=dfTrees.loc[dfTrees['mean_test_r2'].idxmax(),:]

criterion=dfTree.param_criterion
max_features=dfTree.param_max_features
splitter=dfTree.param_splitter
if dfTree.name=='depth_leaf':
    max_depth=dfTree.param_max_depth
    min_samples_leaf=dfTree.param_min_samples_leaf
    tree=DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_features=max_features,max_depth=max_depth,min_samples_leaf=min_samples_leaf,random_state=42)
elif dfTree.name=='alpha':
    ccp_alpha=dfTree.param_ccp_alpha
    tree=DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_features=max_features,ccp_alpha=ccp_alpha,random_state=42)


##############################GRADIENT BOOSTING DESCEND REGRESSOR#################################
print("Train HGBD")
#:Grid 2160 models HistGradient-3min. 486 models Gradient-
# If criterion (MSE or Poisson) predicted value is mean of terminal nodes. If criterion=MAE predicted value is median.
# Prevent overffitting with min_samples_leaf+max_depth or ccp_alpha

# U tell the grid which hyperparam it refers too by setting the name of the transformer (poly) with '__' and the name of the parameter (degree).

###GRIDSEARCHCV###
if criterion not in ['squared_error', 'friedman_mse']:
    criterion = 'friedman_mse'

if (Xtrain.shape[0] >= 10000) and (dfTree.name == "depth_leaf"):  # 10000
    if max_features == None:
        max_features = [0.6, 0.8, 1.0]
    elif max_features == "sqrt":
        max_features = [int(np.sqrt(Xtrain.shape[1]))]
    elif max_features == "log2":
        max_features = [int(np.log2(Xtrain.shape[1]))]
    elif type(max_features) in [float, int]:
        max_features = [float(max_features)]
    else:
        max_features = [0.6, 0.8, 1.0]
    cat_idx = dfX.columns.tolist().index('SteelFamCluster')
    grid = {"categorical_features": [[cat_idx], None], "max_iter": [70, 100, 150], "max_features": max_features,
            'loss': ['squared_error', 'absolute_error', 'gamma', 'poisson'], 'learning_rate': np.logspace(-2, 0, 5),
            'l2_regularization': [0.0] + np.logspace(-2, 2, 5).tolist(),
            'interaction_cst': [None, 'pairwise', 'no_interactions']}  #

    GBRTree = HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=10,max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, random_state=42, verbose=0)
else:
    if max_features == None:
        max_features = [0.6, 0.8, 1.0]
    grid = {"n_estimators": [70, 100, 150], "subsample": [0.8, 0.9, 1.0], "max_features": max_features,
            'loss': ['squared_error', 'absolute_error', 'huber'], 'learning_rate': np.logspace(-2, 0, 5),
            'alpha': np.linspace(0, 0.99, 4).tolist() + [0.9]}  # ,'criterion':['friedman_mse','squared_error']
    if dfTree.name == 'depth_leaf':
        GBRTree = GradientBoostingRegressor(criterion=criterion, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, n_iter_no_change=10, random_state=42,
                                            verbose=0)  # validation_fraction=0.2,init='zero',
    elif dfTree.name == 'alpha':
        GBRTree = GradientBoostingRegressor(criterion=criterion, ccp_alpha=ccp_alpha,
                                            n_iter_no_change=10, random_state=42, verbose=0)

cv = GroupShuffleSplit(n_splits=5, train_size=.8, random_state=42)

# With all features
GridGBRTree = GridSearchCV(estimator=GBRTree, param_grid=grid, scoring=['r2', "neg_root_mean_squared_error"],
                               return_train_score=True, refit='r2', cv=cv, n_jobs=-1, verbose=2)  # error_score='raise',
GridGBRTree.fit(Xtrain, ytrain[:, 0].reshape(1, -1)[0], groups=groups[train_idx],
                    sample_weight=train_sample_weight)
# Serialization
if (Xtrain.shape[0] >= 10000) and (dfTree.name == "depth_leaf"):
    with open("GridHGBRTreeModel.pickle", "wb") as outfile:
        pickle.dump(GridGBRTree, outfile)
else:
    with open("GridGBRTreeModel.pickle", "wb") as outfile:
        pickle.dump(GridGBRTree, outfile)



###TRAIN BEST MODEL###
data1=pd.DataFrame(GridGBRTree.cv_results_).loc[pd.DataFrame(GridGBRTree.cv_results_).rank_test_r2==1,:]
if data1.ndim>1:
    data1=data1.iloc[0,:]
dfGBRTree=data1
dfGBRTree.name = "all"

loss = dfGBRTree.params['loss']
learning_rate = dfGBRTree.params['learning_rate']
max_features = dfGBRTree.params['max_features']

if (Xtrain.shape[0] >= 10000) and (dfTree.name == "depth_leaf"):
    max_iter = dfGBRTree.params['max_iter']
    l2_regularization = dfGBRTree.params['l2_regularization']
    interaction_cst = dfGBRTree.params['interaction_cst']
    categorical_features = dfGBRTree.params['categorical_features']

    GBRTree = HistGradientBoostingRegressor(categorical_features=categorical_features, learning_rate=learning_rate,
                                            loss=loss, max_iter=max_iter, l2_regularization=l2_regularization,
                                            interaction_cst=interaction_cst, early_stopping=True,
                                            max_features=max_features, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, n_iter_no_change=10, random_state=42,
                                            verbose=0)  # ,scoring='neg_root_mean_squared_error'
else:
    subsample = dfGBRTree.params['subsample']
    n_estimators = dfGBRTree.params['n_estimators']
    alpha = dfGBRTree.params['alpha']
    if dfTree.name == 'depth_leaf':
        GBRTree = GradientBoostingRegressor(alpha=alpha, n_estimators=n_estimators, learning_rate=learning_rate,
                                            loss=loss, subsample=subsample, criterion=criterion,
                                            max_features=max_features, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, n_iter_no_change=10, random_state=42,
                                            verbose=2)
    elif dfTree.name == 'alpha':
        GBRTree = GradientBoostingRegressor(alpha=alpha, n_estimators=n_estimators, learning_rate=learning_rate,
                                            loss=loss, subsample=subsample, criterion=criterion,
                                            max_features=max_features, ccp_alpha=ccp_alpha, n_iter_no_change=10,
                                            random_state=42, verbose=2)

GBRTree.fit(X=Xtrain, y=ytrain[:, 0].reshape(1, -1)[0], sample_weight=train_sample_weight)
# Serialization
with open("GBRTreeModel.pickle", "wb") as outfile:
    pickle.dump(GBRTree, outfile)
print("Written object", GBRTree)




