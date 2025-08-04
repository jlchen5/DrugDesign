from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def build_predictive_model(df):
    # 特征矩阵
    X = np.array([np.array(fp) for fp in df['fp']])
    y = df['pIC50'].values
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 模型训练
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 评估
    pred = model.predict(X_test)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred))}")
    return model

def predict_new_molecule(model, smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return model.predict([fp])[0]
