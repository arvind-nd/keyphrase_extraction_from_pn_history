from sklearn import model_selection

df = pd.read_csv("../input/train_3.csv")
df["kfold"] = -1
set_seed(args.seed)
Fold = model_selection.GroupKFold(n_splits=args.n_splits)
groups = df["pn_num"].values
for fold_, (_,v_) in enumerate(Fold.split(X=df,y=df["location"],groups=groups)):
    df.loc[v_,"kfold"] = fold_
df.to_csv("../input/train_5folds.csv", index=False)
