model_dic = {"BNB": {"Accuracy": 1, "Precision": 2, "Recall": 3, "F1": 4},
             "LR":{"Accuracy": 11, "Precision": 22, "Recall": 33, "F1": 44},
             "SVM":{"Accuracy": 111, "Precision": 222, "Recall": 333, "F1": 444},
             "RF":{"Accuracy": 1111, "Precision": 2222, "Recall": 3333, "F1": 4444}}


max_precision = max(values["Precision"] for key, values in model_dic.items())
print(max_precision)
model = [model for model, precision in model_dic.items() if precision["Precision"] == max_precision]
print(model)

print(type(model))
print(model[0])
print(type(model[0]))

model_best_f1 = ["LR", "RF"]

# subset the model_dic to be the ones have the highest F1 Score
precision_dic = {key: value for key, value in model_dic.items() if key in model_best_f1}

print(precision_dic)


