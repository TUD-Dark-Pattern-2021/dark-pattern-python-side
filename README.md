# Detection_Pipeline

The 'application.py' is the detection process, including both dark pattern detection to dark pattern category classification.

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

# Usage
run ``` python Pipeline.py ```

## Api: 
### name: ``` /api/parse ```
### params @dict
* content: Array*
* key: Array
### Return @json
eg:
```json
{
    "items_counts": {
        "3": 1
    },
    "details": [
        {
            "content": "LAST CHANCE ITEMS",
            "key": "AKIAX4DWMQK2GILTOxx",
            "category": 3,
            "category_name": "Scarcity"
        }
    ]
}
```

Prerequisite & dependencies
Python 3.9
Pip
[Awsebcli](https://docs.aws.amazon.com/zh_cn/elasticbeanstalk/latest/dg/eb-cli3-install-windows.html)











