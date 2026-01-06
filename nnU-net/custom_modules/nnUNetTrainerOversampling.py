import sys
import time
import json
import os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

# Import the custom dataloader from the same directory
# We assume this file is in nnU-net/custom_modules/
try:
    from custom_dataloader import CustomOversamplingDataLoader
except ImportError:
    # If installed as a package or running from elsewhere, try relative import
    from .custom_dataloader import CustomOversamplingDataLoader


class nnUNetTrainerOversampling(nnUNetTrainer):
    """
    Custom nnU-Net trainer with:
    - Oversampling for rare subjects (loaded from JSON)
    - Frequent checkpoint saving
    - Time-limit safety for Kaggle
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        # Initialize parent without unpack_dataset
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # -------------------------------------------------------
        # Load Configuration from JSON
        # -------------------------------------------------------
        self.rare_subjects = []
        self.oversample_factor = 3.0
        
        # Locate rare_subjects.json relative to this file
        current_dir = Path(__file__).parent
        json_path = current_dir / "rare_subjects.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.rare_subjects = json.load(f)
            print(f"Loaded {len(self.rare_subjects)} rare subjects from {json_path}")
        else:
            print(f"⚠ Warning: rare_subjects.json not found at {json_path}. Oversampling disabled.")

        # SAVE FREQUENCY: save permanent checkpoint every 20 epochs
        self.save_every = 20

        # TIME LIMIT: stop training after ~11h (Kaggle max is ~12h)
        # Defaulting to 11h 45m if not overwritten
        self.max_time_seconds = (11 * 60 + 45) * 60
        self.start_time = time.time()

    # -------------------------------------------------------
    # Oversampling Logic
    # -------------------------------------------------------
    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        return dataset_tr, dataset_val

    def get_plain_dataloaders(self):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = CustomOversamplingDataLoader(
            dataset_tr,
            self.batch_size,
            self.patch_size,
            self.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            rare_subjects=self.rare_subjects,
            oversample_factor=self.oversample_factor,
        )

        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.patch_size,
            self.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
        )

        return dl_tr, dl_val

    # -------------------------------------------------------
    # Time-Limit Safety
    # -------------------------------------------------------
    def on_epoch_end(self):
        super().on_epoch_end()

        elapsed = time.time() - self.start_time

        if elapsed > self.max_time_seconds:
            self.print_to_log_file(
                f"\\nTIME LIMIT REACHED ({elapsed / 3600:.2f} hours)."
            )
            self.print_to_log_file(
                "Stopping training gracefully to save checkpoints."
            )

            # Explicitly save latest checkpoint
            self.save_checkpoint(
                join(self.output_folder, "checkpoint_latest.pth")
            )

            # Clean shutdown so Kaggle persists outputs
            self.on_train_end()
            sys.exit(0)
