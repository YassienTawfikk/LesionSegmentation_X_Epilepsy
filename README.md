# LesionSegmentation_X_Epilepsy

This project aims to develop and evaluate advanced deep learning models for automated lesion segmentation across the full spectrum of epilepsy. By leveraging diverse
architectures and multimodal MRI data, the goal is to enhance segmentation accuracy and robustness. Ultimately, this work seeks to improve clinical presurgical
evaluation to support better outcomes in epilepsy surgery.

## Project Structure

```
LesionSegmentation_X_Epilepsy/
├── README.md
├── .gitignore
├── requirements.txt
└── src/
    └── notebooks/
        ├── data_setup/
        │   └── bids_to_centric.ipynb
        │
        └── nnUNet_training/
            ├── FLAIR/
            │   ├── preprocessing/
            │   ├── models/
            │   └── notes.txt
            │
            ├── T1/
            │   ├── preprocessing/
            │   ├── models/
            │   └── notes.txt
            │
            └── T1_FLAIR/
                ├── preprocessing/
                ├── models/
                └── notes.txt
```