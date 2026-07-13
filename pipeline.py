import csv
from models import CSVLoader
from scorer import StatisticalScorer
from refiner import WeightRefiner

class BotDetectorPipeline:
    def __init__(self, input_csv: str, output_csv: str):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.loader = CSVLoader(input_csv)
        self.scorer = StatisticalScorer()
        self.refiner = WeightRefiner()

    def run(self):
        print(f"[*] Loading data from {self.input_csv}...")
        records = self.loader.load()
        print(f"[*] Loaded {len(records)} records.")

        print("[*] Stage 1: Running statistical scoring...")
        stage1_results = self.scorer.score_population(records)

        print("[*] Stage 2: Refining weights with PCA + Logistic Regression...")
        final_results = self.refiner.refine(stage1_results)

        print(f"[*] Saving results to {self.output_csv}...")
        self._save_results(final_results)
        print("[+] Done!")

    def _save_results(self, results):
        if not results:
            return

        # Prepare headers: ID, Scores, then all features and anomalies
        sample = results[0]
        fieldnames = ['channel_id', 'suspicion_score', 'refined_score', 'risk_band']
        
        # Add feature and anomaly names
        feat_names = [f"feat_{k}" for k in sample['features'].keys()]
        anom_names = [f"anom_{k}" for k in sample['anomalies'].keys()]
        fieldnames.extend(feat_names)
        fieldnames.extend(anom_names)

        with open(self.output_csv, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for res in results:
                row = {
                    'channel_id': res['channel_id'],
                    'suspicion_score': res['suspicion_score'],
                    'refined_score': res['refined_score'],
                    'risk_band': res['risk_band']
                }
                # Flatten features and anomalies
                for k, v in res['features'].items():
                    row[f"feat_{k}"] = round(v, 6) if isinstance(v, float) else v
                for k, v in res['anomalies'].items():
                    row[f"anom_{k}"] = round(v, 6) if isinstance(v, float) else v
                
                writer.writerow(row)

if __name__ == "__main__":
    import sys
    # Use the provided file or default to sample
    input_file = "youtube_sample_100k.csv"
    output_file = "bot_detection_results.csv"
    
    pipeline = BotDetectorPipeline(input_file, output_file)
    pipeline.run()
