# LesionSegmentation_X_Epilepsy

This project aims to develop and evaluate advanced deep learning models for automated lesion segmentation across the full spectrum of epilepsy. By leveraging diverse architectures and multimodal MRI data, the goal is to enhance segmentation accuracy and robustness. Ultimately, this work seeks to improve clinical presurgical evaluation to support better outcomes in epilepsy surgery.

---

## Supervisors

**Automated Lesion Segmentation for Pre-Surgical Evaluation in Epilepsy**
*Sep 2025 – Present*

Under the supervision of:

* **Dr. Aya Fawzy Khalaf**
  Associate Research Scientist at Yale University in the Blumenfeld Lab.
  [Yale Profile](https://medicine.yale.edu/profile/aya-khalaf/)

* **Eng. Mahmoud Salman**
  MESc Biomedical Engineering student at Western University.
  [LinkedIn Profile](https://linkedin.com/in/mahmoud1yaser)

* **Dr. Tamer Basha**
  Associate Professor at Cairo University and Postdoctoral Fellow at Harvard Medical School.
  [LinkedIn Profile](https://www.linkedin.com/in/tamer-basha-b81812ab/)

---

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
        ├── SynthSeg/
        │   └── launch_synthSeg.ipynb
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
