import numpy as np
from collections import defaultdict

class WeightEstimator:
    """
    Estimates bird weight using bounding box dimensions as proxy.
    Weight proxy = bbox_area * height_factor
    
    To convert to actual grams, calibration data is needed:
    weight_grams = a * weight_proxy + b (linear regression)
    """
    
    def __init__(self, calibration_data=None):
        """
        Args:
            calibration_data: dict with 'a' and 'b' for linear conversion
                             If None, returns raw proxy values
        """
        self.calibration_data = calibration_data
        self.track_weights = defaultdict(list)  # Store weights per track_id
    
    def calculate_weight_proxy(self, bbox):
        """
        Calculate weight proxy from bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] in pixels
            
        Returns:
            weight_proxy: float (unitless, needs calibration for grams)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Bounding box area (pixels²)
        area = width * height
        
        # Height factor (normalized) - taller birds appear closer/larger
        height_factor = height / 100.0
        
        # Weight proxy: area weighted by height
        weight_proxy = area * height_factor
        
        return weight_proxy
    
    def estimate_weight(self, bbox):
        """
        Estimate weight with optional calibration to grams
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            weight: float (grams if calibrated, else proxy value)
            unit: str ('g' or 'proxy')
        """
        proxy = self.calculate_weight_proxy(bbox)
        
        if self.calibration_data:
            # Convert proxy to grams: weight = a * proxy + b
            a = self.calibration_data['a']
            b = self.calibration_data['b']
            weight = a * proxy + b
            return weight, 'g'
        else:
            return proxy, 'proxy'
    
    def update_track_weight(self, track_id, bbox):
        """
        Store weight measurements for a tracked bird
        """
        weight, _ = self.estimate_weight(bbox)
        self.track_weights[track_id].append(weight)
    
    def get_average_weight(self, track_id):
        """
        Get average weight for a tracked bird (reduces noise)
        """
        if track_id not in self.track_weights:
            return None
        weights = self.track_weights[track_id]
        return np.mean(weights)
    
    def get_all_weights_summary(self):
        """
        Get summary statistics for all tracked birds
        """
        all_weights = []
        per_bird = []
        
        for track_id, weights in self.track_weights.items():
            avg_weight = np.mean(weights)
            all_weights.append(avg_weight)
            per_bird.append({
                "id": int(track_id),
                "weight_proxy": round(float(avg_weight), 2),
                "samples": len(weights),
                "std_dev": round(float(np.std(weights)), 2)
            })
        
        unit = 'g' if self.calibration_data else 'proxy'
        
        return {
            "unit": unit,
            "avg_weight": round(float(np.mean(all_weights)), 2) if all_weights else 0,
            "min_weight": round(float(np.min(all_weights)), 2) if all_weights else 0,
            "max_weight": round(float(np.max(all_weights)), 2) if all_weights else 0,
            "per_bird": sorted(per_bird, key=lambda x: x['id']),
            "total_birds": len(per_bird),
            "calibration_needed": not bool(self.calibration_data)
        }


# Example usage and calibration info
if __name__ == "__main__":
    # Without calibration (proxy values)
    estimator = WeightEstimator()
    
    # Example bounding box [x1, y1, x2, y2]
    test_bbox = [100, 150, 200, 250]
    
    weight, unit = estimator.estimate_weight(test_bbox)
    print(f"Weight proxy: {weight:.2f} ({unit})")
    
    # Simulate tracking multiple frames
    estimator.update_track_weight(track_id=1, bbox=[100, 150, 200, 250])
    estimator.update_track_weight(track_id=1, bbox=[102, 148, 198, 252])
    estimator.update_track_weight(track_id=2, bbox=[300, 200, 380, 290])
    
    summary = estimator.get_all_weights_summary()
    print("\n Weight Summary:")
    print(f"Average weight: {summary['avg_weight']} {summary['unit']}")
    print(f"Total birds tracked: {summary['total_birds']}")
    print(f"Calibration needed: {summary['calibration_needed']}")
    
    print("\n Calibration Requirements:")
    print("To convert proxy to actual grams, you need:")
    print("1. Sample videos with known bird weights (ground truth)")
    print("2. Calculate weight_proxy for each bird")
    print("3. Linear regression: weight_grams = a * proxy + b")
    print("4. Example: If proxy=10000 → 1500g, proxy=15000 → 2000g")
    print("   Then: a = 0.1, b = 500")