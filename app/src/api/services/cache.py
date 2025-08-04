"""
Simple in-memory cache for models and predictions.
No Redis needed - just Python dicts with timestamps.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("cache")


class ModelCache:
    """Simple in-memory cache for trained models."""
    
    def __init__(self, cache_ttl_minutes: int = 60):
        """
        Initialize cache.
        
        Args:
            cache_ttl_minutes: Time to live for cached predictions
        """
        self.models: Dict[int, Any] = {}  # part_id -> model
        self.predictions: Dict[str, Any] = {}  # cache_key -> (prediction, timestamp)
        self.model_metadata: Dict[int, Dict] = {}  # part_id -> metadata
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.logger = logger
        
    def load_all_models(self):
        """Load all trained models from checkpoint directory."""
        checkpoint_dir = Path(__file__).parent.parent.parent.parent / "checkpoints"
        
        if not checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return
            
        # Load training results for metadata
        results_file = checkpoint_dir / "training_results_final.csv"
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            
            for _, row in results_df.iterrows():
                if row['status'] == 'success':
                    part_id = int(row['part_id'])
                    self.model_metadata[part_id] = {
                        'stock_code': row.get('stock_code'),
                        'mae': row.get('mae'),
                        'zero_pct': row.get('zero_pct'),
                        'model_path': row.get('model_path')
                    }
        
        # Load actual models
        model_files = list(checkpoint_dir.glob("model_part_*.pkl"))
        
        for model_file in model_files:
            try:
                # Extract part_id from filename
                part_id_str = model_file.stem.replace("model_part_", "").split(".")[0]
                part_id = int(part_id_str)
                
                # Load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                self.models[part_id] = model
                self.logger.info(f"Loaded model for part {part_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_file}: {e}")
                
        self.logger.info(f"Loaded {len(self.models)} models into cache")
        
    def get_model(self, part_id: int) -> Optional[Any]:
        """Get model from cache."""
        return self.models.get(part_id)
        
    def get_model_metadata(self, part_id: int) -> Optional[Dict]:
        """Get model metadata."""
        return self.model_metadata.get(part_id, {})
        
    def cache_prediction(self, cache_key: str, prediction: Any):
        """Cache a prediction with timestamp."""
        self.predictions[cache_key] = (prediction, datetime.now())
        
    def get_cached_prediction(self, cache_key: str) -> Optional[Any]:
        """Get cached prediction if still valid."""
        if cache_key in self.predictions:
            prediction, timestamp = self.predictions[cache_key]
            
            # Check if still valid
            if datetime.now() - timestamp < self.cache_ttl:
                return prediction
            else:
                # Remove expired cache
                del self.predictions[cache_key]
                
        return None
        
    def clear_expired_predictions(self):
        """Remove expired predictions from cache."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.predictions.items()
            if now - timestamp >= self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.predictions[key]
            
        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired predictions")
            
    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size information."""
        return {
            'models': len(self.models),
            'predictions': len(self.predictions),
            'metadata': len(self.model_metadata)
        }
        
    def clear(self):
        """Clear all caches."""
        self.models.clear()
        self.predictions.clear()
        self.model_metadata.clear()
        self.logger.info("Cache cleared")
        
    def get_loaded_parts(self) -> List[int]:
        """Get list of part IDs with loaded models."""
        return list(self.models.keys())