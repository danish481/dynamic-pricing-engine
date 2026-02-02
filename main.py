from src.data_engine import OlistDataEngine
from src.feature_generator import FeatureGenerator
from src.experiments import RegressionMastery
import joblib

# 1. ETL: Load 9 Datasets & Merge
engine = OlistDataEngine()
engine.get_paths()
datasets = engine.load_data()
master_table = engine.construct_master_table()

# 2. FE: Engineer Complex Features
gen = FeatureGenerator(master_table)
final_df = gen.preprocess_and_engineer()

# 3. ML: Run Experiments (VIF + SGD + Error Analysis)
experiment = RegressionMastery(final_df)
model, scaler = experiment.run_production_pipeline()

# 4. Save Artifacts for App
print("\n💾 Saving Production Model...")
joblib.dump(model, 'src/best_model.pkl')
joblib.dump(scaler, 'src/best_scaler.pkl')
print("✅ Done.")