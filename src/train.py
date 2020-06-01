import pandas as pd
from sklearn import preprocessing
import os
from sklearn import ensemble, metrics

TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD = int(os.environ.get('FOLD'))

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [1,2,3,4],
    2: [1,2,3,4],
    3: [1,2,3,4],
    4: [1,2,3,4]
}


if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis =1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis =1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_tf.lloc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))

    #data is ready to train
    clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2)
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1 ]
    print(preds) 
    print(metrics.roc_auc_score(yvalid, preds))