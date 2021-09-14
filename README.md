# Detection_Pipeline

The 'Pipeline.py' is the detection process, from web scrapping --- dark pattern detection --- dark pattern category classification.

## Packages needed for the file to run:

(1) `pandas`

(2) `numpy`

(3) `selenium`

(4) `lxml.html`

(5) `re`

(6) `time`

(7) `csv`

(8) `joblib`

## Pretrained models used in the process

(1) `Presence model`, to detect if a line of the content contains dark pattern.

`bnb_presence_classifier.joblib` is the presence model.

`presence_CountVectorizer.joblib` is the presence countverctorizer for content preprocessing.

(2) `Category model`, to classify the detected dark pattern content into certain dark pattern type.

`mnb_category_classifier.joblib` is the category model.

`category_CountVectorizer.joblib` is the category countverctorizer for content preprocessing.













