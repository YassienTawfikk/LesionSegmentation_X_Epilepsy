import os
import sys

# ---------------------------------------------------------
# 1. Base Paths
# ---------------------------------------------------------
# Define the root directory for data relative to this config file
# This assumes config.py is in nnU-net/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------
# 2. nnU-Net Environment Variables
# ---------------------------------------------------------
nnUNet_raw = os.path.join(DATA_ROOT, "nnUNet_raw_data_base", "nnUNet_raw")
nnUNet_preprocessed = os.path.join(DATA_ROOT, "nnUNet_preprocessed")
nnUNet_results = os.path.join(DATA_ROOT, "nnUNet_results")

# ---------------------------------------------------------
# 3. Project Specific Data Paths
# ---------------------------------------------------------
RAW_DATA_PATH = os.path.join(DATA_ROOT, "bonn_fcd_fixed")
PARTICIPANTS_DATA_DIR = os.path.join(DATA_ROOT, "participants-data")
EXCEL_PATH = os.path.join(PARTICIPANTS_DATA_DIR, "participants.xlsx")
SPLITS_FILE_PATH = os.path.join(BASE_DIR, "splits_final.json")

# ---------------------------------------------------------
# 4. Training Parameters
# ---------------------------------------------------------
TRAINING_TIME_MINUTES = (11 * 60) + 45  # 11 hours 45 minutes
OVERSAMPLE_FACTOR = 3.0

RARE_SUBJECTS = [
    'sub-00001', 'sub-00003', 'sub-00014', 'sub-00015', 'sub-00016', 'sub-00018', 
    'sub-00024', 'sub-00027', 'sub-00033', 'sub-00038', 'sub-00040', 'sub-00044', 
    'sub-00050', 'sub-00053', 'sub-00055', 'sub-00058', 'sub-00060', 'sub-00063', 
    'sub-00065', 'sub-00073', 'sub-00077', 'sub-00078', 'sub-00080', 'sub-00081', 
    'sub-00083', 'sub-00087', 'sub-00089', 'sub-00097', 'sub-00098', 'sub-00101', 
    'sub-00105', 'sub-00109', 'sub-00112', 'sub-00115', 'sub-00116', 'sub-00122', 
    'sub-00123', 'sub-00126', 'sub-00130', 'sub-00132', 'sub-00133', 'sub-00138', 
    'sub-00146'
]

# ---------------------------------------------------------
# 5. Helper Functions
# ---------------------------------------------------------
def setup_env():
    """Sets up os.environ variables for nnU-Net"""
    os.environ['nnUNet_raw'] = nnUNet_raw
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
    os.environ['nnUNet_results'] = nnUNet_results
    # os.environ['nnUNet_compile'] = 'false' # Uncomment if needed locally
    
    print(f"✅ nnU-Net environment variables set from config.py")
    print(f"   RAW: {nnUNet_raw}")
    print(f"   PREPROCESSED: {nnUNet_preprocessed}")
    print(f"   RESULTS: {nnUNet_results}")

def add_custom_modules_to_path():
    """Adds custom_nnunet directory to python path"""
    custom_dir = os.path.join(BASE_DIR, "custom_nnunet")
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)
        print(f"✅ Added {custom_dir} to sys.path")
