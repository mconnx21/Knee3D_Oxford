# Knee3D_Oxford
This code enables registration of 2D x-rays to segmented MRI volumes using pseudo-xrays.

## Set up

1. Create an environment using requirements.txt

2. Create the following folders in the root directory:

- csvs
- data
- data_for_report_plots
- masks
- MLSegments
- oai
- report
- results
- xrays

3. Add data.

Irina has been given the core folders of data. Copy the MLSegments folder, the oai folder and the xrays folder into this repository.

Sources: 
- https://www.kaggle.com/datasets/kgaooo/oai-tissue-segmentations. 
- OAI dataset


4. Pick a similarity measure and experiment type at the bottom of intensity_registration.py, and run the file.

You can visually analyze results by patient in the results folder.
You can see quantative data results in the csvs folder.



