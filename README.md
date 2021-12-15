# Detection_Pipeline

The 'application.py' is the detection process, including both dark pattern detection to dark pattern type classification.

## Packages needed for the file to run:

Can be found in the file: "requirements.txt"

## Pretrained models used in the process

(1) `Dark Pattern Detection model`, to detect if a line of the content contains dark pattern.

`rf_presence_classifier.joblib` is the 5 dark pattern types detection model --- Random Forest Model

`presence_TfidfVectorizer.joblib` is the presence countverctorizer for content preprocessing for the 5 Pattern Types.

`confirm_rf_clf.joblib` is the Confirmshaming dark pattern detection model --- Random Forest Model

`confirm_tv.joblib` is the presence countverctorizer for content preprocessing for Confimshaming.

(2) `Type Classification model`, to classify the detected dark pattern content into certain dark pattern type.

`lr_category_classifier.joblib` is the pattern type classification model ---- Logistic Regression Model

`type_CountVectorizer.joblib` is the pattern type countverctorizer for content preprocessing.

# Usage
run ``` python application.py ```

## Api: 
### name: ``` /api/parse ```
### params @dict
* content: Array*
* key: Array
### Return @json
eg:
```json
{
    "total_counts": {1},
    "items_counts": {
        "1": 1
    },
    "details": [
        {
            "content": "LAST CHANCE ITEMS",
            "tag": "div/div/p/div",
            "tag_type": "link",
            "key": "AKIAX4DWMQK2GILTOxx",
            "type_name": "FakeCountdown",
            "type_name_slug": "Fake Countdown"
        }
    ]
}
```

Prerequisite & dependencies
Python 3.9
Pip
[Awsebcli](https://docs.aws.amazon.com/zh_cn/elasticbeanstalk/latest/dg/eb-cli3-install-windows.html)











