def main():
    # 数据准备
    active_data = load_chembl_data("chembl_data.csv")
    df = featurize_molecules(active_data)
    
    # 模型构建
    model = build_predictive_model(df)
    
    # 新分子预测
    new_smiles = "CCOc1ccccc1NC(=O)C"
    predicted_ic50 = predict_new_molecule(model, new_smiles)
    print(f"Predicted pIC50: {predicted_ic50:.2f}")
    
    # 分子对接验证
    prepare_receptor("target_protein.pdb")
    docking_score = run_vina_docking(new_smiles, "docking_results")
    print(f"Docking Affinity: {docking_score} kcal/mol")

if __name__ == "__main__":
    main()
