"""Predict late deliveries from order-time features; never uses actual delivery as input."""
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np,joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True);p.add_argument("--ordered",default="order_date");p.add_argument("--promised",default="promised_date");p.add_argument("--delivered",default="actual_delivery_date");p.add_argument("--supplier",default="supplier_id");p.add_argument("--out",default="outputs");a=p.parse_args()
 d=pd.read_csv(a.input)
 for c in [a.ordered,a.promised,a.delivered]:d[c]=pd.to_datetime(d[c],errors="coerce")
 d=d.dropna(subset=[a.ordered,a.promised,a.delivered]).sort_values(a.ordered).copy();d["days_promised"]=(d[a.promised]-d[a.ordered]).dt.days;d["late"]=(d[a.delivered]>d[a.promised]).astype(int);d["order_month"]=d[a.ordered].dt.month;d["order_weekday"]=d[a.ordered].dt.dayofweek
 y=d.late;X=d.drop(columns=["late",a.delivered,a.promised,a.ordered]);cut=int(len(d)*.8);tr,te=d.iloc[:cut],d.iloc[cut:]
 num=X.select_dtypes(include="number").columns;cat=X.columns.difference(num)
 prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
 m=Pipeline([("features",prep),("model",LogisticRegression(max_iter=1200,class_weight="balanced"))]);m.fit(X.iloc[:cut],y.iloc[:cut]);pr=m.predict_proba(X.iloc[cut:])[:,1]
 metrics={"rows":len(d),"train_rows":len(tr),"test_rows":len(te),"late_rate":float(y.iloc[cut:].mean()),"roc_auc":float(roc_auc_score(y.iloc[cut:],pr)),"average_precision":float(average_precision_score(y.iloc[cut:],pr)),"brier":float(brier_score_loss(y.iloc[cut:],pr)),"split":"last 20% by order_date"}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(metrics,indent=2));pd.DataFrame({"order_row":d.index[cut:],"supplier":d[a.supplier].iloc[cut:].values,"late_probability":pr,"actual_late":y.iloc[cut:].values}).to_csv(out/"late_risk_queue.csv",index=False);joblib.dump({"model":m,"features":X.columns.tolist()},out/"model.joblib");print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
