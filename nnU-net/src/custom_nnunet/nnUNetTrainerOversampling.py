import sys
import time
import os
from batchgenerators.utilities.file_and_folder_operations import join
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom imports
# Helper to find sibling module if not in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from custom_dataloader import CustomOversamplingDataLoader
from config import RARE_SUBJECTS, OVERSAMPLE_FACTOR, TRAINING_TIME_MINUTES

class nnUNetTrainerOversampling(nnUNetTrainer):
    """
    Custom nnU-Net trainer with:
    - Oversampling for rare subjects
    - Frequent checkpoint saving
    - Time-limit safety for Kaggle/Local limits
    """
    
    # -------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------
    rare_subjects = RARE_SUBJECTS
    oversample_factor = OVERSAMPLE_FACTOR
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        # Initialize parent
        # Note: unpack_dataset argument removed for v2 compat if needed, following notebook Ref
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        
        # 🔥 SAVE FREQUENCY: Save a permanent checkpoint every 20 epochs
        self.save_every = 20
        
        # 🔥 TIME LIMIT: Stop training after limit
        self.max_time_seconds = TRAINING_TIME_MINUTES * 60 
        self.start_time = time.time()

    # -------------------------------------------------------
    # 2. Oversampling Logic
    # -------------------------------------------------------
    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        return dataset_tr, dataset_val
    
    def get_plain_dataloaders(self):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        dl_tr = CustomOversamplingDataLoader(
            dataset_tr, self.batch_size, self.patch_size, self.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            rare_subjects=self.rare_subjects, oversample_factor=self.oversample_factor
        )
        
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, self.patch_size, self.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent
        )
        return dl_tr, dl_val

    # -------------------------------------------------------
    # 3. Time Check Logic
    # -------------------------------------------------------
    def on_epoch_end(self):
        """Check time at the end of every epoch"""
        super().on_epoch_end()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # If we passed the limit, force a stop
        if elapsed > self.max_time_seconds:
            self.print_to_log_file(f"\\n⏰ TIME LIMIT REACHED ({elapsed/3600:.2f} hours).")
            self.print_to_log_file("Stopping training gracefully to save checkpoints safely.")
            
            # Force nnU-Net to save the latest state explicitly
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            
            # Stop the training loop
            self.on_train_end()
            sys.exit(0) # Exit the script successfully
