import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class WeightRefiner:
    def __init__(self, top_k_percent: float = 0.05, bottom_k_percent: float = 0.30):
        # Adjusted defaults: look for cleaner separations at the true extremes
        self.top_k = top_k_percent
        self.bottom_k = bottom_k_percent
        
        self.scaler = None
        self.pca = None
        self.clf = None

    def refine(self, stage1_results: List[Dict]) -> List[Dict]:
        if not stage1_results:
            return stage1_results

        # Prepare data matrix from anomalies
        feature_names = sorted(stage1_results[0]['anomalies'].keys())
        X = []
        scores = []
        for res in stage1_results:
            X.append([res['anomalies'][fn] for fn in feature_names])
            scores.append(res['suspicion_score'])
        
        X = np.array(X)
        scores = np.array(scores)

        # Scale and run PCA on the ENTIRE population to learn true global variance
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Keep components explaining 90% of variance globally
        self.pca = PCA(n_components=0.9, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # Pseudo-labeling (Find clear separators at the extreme tails)
        threshold_high = np.percentile(scores, 100 * (1 - self.top_k))
        threshold_low = np.percentile(scores, 100 * self.bottom_k)
        
        y_pseudo = np.zeros(len(scores))
        y_pseudo[scores >= threshold_high] = 1
        
        # Isolate training subset from the global PCA matrix
        mask = (scores >= threshold_high) | (scores <= threshold_low)
        X_train = X_pca[mask]
        y_train = y_pseudo[mask]

        # SVM with an RBF kernel handles complex, non-linear anomaly boundaries
        self.clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        self.clf.fit(X_train, y_train)

        # Re-evaluate all samples using the non-linear decision space
        # Yields a calibrated probability map based on non-linear proximity to bot clusters
        refined_probs = self.clf.predict_proba(X_pca)[:, 1]

        # Merge results
        for i, res in enumerate(stage1_results):
            res['refined_score'] = round(float(refined_probs[i]), 4)
            res['risk_band'] = self._get_risk_band(res['refined_score'])

        return stage1_results

    def _get_risk_band(self, score: float) -> str:
        if score >= 0.85: return "High"
        if score >= 0.60: return "Elevated"
        if score >= 0.25: return "Low"
        return "Minimal"
    
    def refine_single(self, stage1_single_result: Dict) -> Dict:
        """Refines a single influencer's score using the pre-trained Scaler, PCA, and SVM."""
        # Ensure the model was actually trained first
        if not hasattr(self, 'scaler') or not hasattr(self, 'pca') or not hasattr(self, 'clf'):
            raise ValueError("Refiner must be trained on the global population before scoring single records.")
            
        feature_names = sorted(stage1_single_result['anomalies'].keys())
        x = np.array([[stage1_single_result['anomalies'][fn] for fn in feature_names]])
        
        # Apply the exact same global transforms without refitting
        x_scaled = self.scaler.transform(x)
        x_pca = self.pca.transform(x_scaled)
        
        # Predict probability using the RBF decision boundary
        refined_prob = self.clf.predict_proba(x_pca)[0, 1]
        
        stage1_single_result['refined_score'] = round(float(refined_prob), 4)
        stage1_single_result['risk_band'] = self._get_risk_band(stage1_single_result['refined_score'])
        
        return stage1_single_result