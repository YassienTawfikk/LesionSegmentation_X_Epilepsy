from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
import numpy as np
import sys
import os

# Ensure config can be imported if running from this file directly or sub-module
# Assuming layout: nnU-net/custom_nnunet/custom_dataloader.py
# and nnU-net/config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import RARE_SUBJECTS, OVERSAMPLE_FACTOR
except ImportError:
    # Fallback if config not found (e.g. strict environment), though we expect it to be there.
    print("Warning: config.py not found from custom_dataloader.py, using defaults")
    RARE_SUBJECTS = []
    OVERSAMPLE_FACTOR = 3.0

class CustomOversamplingDataLoader(nnUNetDataLoader):
    """Custom DataLoader with oversampling for both 2D and 3D"""
    
    def __init__(self, data, batch_size, patch_size, final_patch_size=None,
                 label_manager=None, oversample_foreground_percent=0.33,
                 sampling_probabilities=None, pad_kwargs_data=None,
                 pad_mode="constant", rare_subjects=None, oversample_factor=None):
        
        self.rare_subjects = rare_subjects if rare_subjects is not None else RARE_SUBJECTS
        self.oversample_factor = oversample_factor if oversample_factor is not None else OVERSAMPLE_FACTOR
        
        # Initialize parent class WITHOUT num_threads_in_multithreaded
        super().__init__(data, batch_size, patch_size, final_patch_size,
                        label_manager, oversample_foreground_percent,
                        sampling_probabilities, pad_kwargs_data, pad_mode)
        
        # Modify sampling probabilities after initialization
        if self.rare_subjects:
            self._modify_sampling_probabilities()
    
    def _modify_sampling_probabilities(self):
        """Modify sampling probabilities to favor rare subjects"""
        if not hasattr(self, '_data') or self._data is None:
            return
        
        # Get the list of case identifiers from the dataset
        # nnUNetDatasetBlosc2 uses the 'identifiers' attribute
        case_ids = None
        
        # Try the identifiers attribute (works for nnUNetDatasetBlosc2)
        if hasattr(self._data, 'identifiers'):
            case_ids = self._data.identifiers
        
        # Fallback: try other possible attributes
        if case_ids is None and hasattr(self._data, 'indices'):
            case_ids = self._data.indices
        
        if case_ids is None and hasattr(self._data, 'keys') and callable(self._data.keys):
            try:
                case_ids = list(self._data.keys())
            except:
                pass
        
        if case_ids is None:
            print("  ⚠️ Warning: Could not extract case IDs from dataset, oversampling disabled")
            return
        
        num_cases = len(case_ids)
        sampling_probs = np.ones(num_cases)
        
        # Increase weight for rare subjects
        rare_count = 0
        for idx, case_id in enumerate(case_ids):
            # Extract subject ID (format: sub-00001 or sub-00001_0000)
            subject_id = str(case_id).split('_')[0]
            
            if subject_id in self.rare_subjects:
                sampling_probs[idx] *= self.oversample_factor
                rare_count += 1
        
        # Normalize probabilities
        sampling_probs = sampling_probs / sampling_probs.sum()
        self.sampling_probabilities = sampling_probs
        
        print(f"  ✓ Modified sampling: {rare_count}/{num_cases} cases are rare subjects ({self.oversample_factor}x weight)")
