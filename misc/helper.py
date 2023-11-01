
def confusionMatrixOfClassify(test_loader, model, classes):
  
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    import numpy as np
    ground_truth = []
    predict=[]
    labels = list(range(classes))  # Generate a list from 0 to classes-1


    for batch_idx, pack in enumerate(test_loader):
        img = pack['img'] #[16, 3, 512, 512]
        cls_label = pack['label'] #[16, ]
        cls_label = cls_label.cpu().numpy()
        out = model(img)
        y_predict = np.argmax(out.detach().cpu().numpy(), axis=1)  # Predicted labels for the batch
        
        ground_truth.append(cls_label)
        predict.append(y_predict)

    # After iterating through all batches
    ground_truth = np.concatenate(ground_truth, axis=0)  # Convert the list to a single array
    predict = np.concatenate(predict, axis=0) 
    cm = confusion_matrix(ground_truth, predict,labels=labels)
    print(cm)
    report(ground_truth, predict)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    cm_display.plot()


def report(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import confusion_matrix


    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # Extract TP, FN, TN, FP for each class
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    FP = np.sum(cm, axis=0) - TP
    TN = np.sum(cm) - (TP + FN + FP) # number - list = list
    # print(np.sum(cm, axis=1)) 
    # print(np.sum(cm))
    # print('TP', TP)
    # print('FN', FN)
    # print('FP', FP)
    # print('TN', TN)

    # Calculate sensitivity (recall) for each class
    sensitivity = TP / (TP + FN)

    # Calculate specificity for each class
    specificity = TN / (TN + FP)

    # Calculate precision for each class
    precision = TP / (TP + FP)

    # Calculate accuracy
    accuracy = np.sum(TP) / np.sum(cm)

    # Calculate F1 score for each class
    f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

    # Print the results for each class
    num_classes = len(cm)
    for class_idx in range(num_classes):
        print("Class", class_idx)
        print("Sensitivity (Recall):", sensitivity[class_idx])
        print("Specificity:", specificity[class_idx])
        print("Precision:", precision[class_idx])
        print("F1 Score:", f1_score[class_idx])
        print()

    # Print overall accuracy
    print("Accuracy:", accuracy)