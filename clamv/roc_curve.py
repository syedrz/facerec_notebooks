import numpy as np
import scipy

from sklearn.metrics import roc_curve

def construct_multiclass_roc_curve(y_true, y_pred, num_classes):
	roc_classes = []
	for i in range(num_classes):
		fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=i)

		roc_classes.append([fpr, tpr])

	# Now take all fpr values
	all_fpr = []
	for p in roc_classes:
		all_fpr = np.vstack(p[0])

	all_fpr = np.unique(all_fpr)

	mean_tpr = np.zeros_like(all_fpr)

	for i in range(num_classes):
		mean_tpr += scipy.interp(all_fpr, roc_classes[i][0], roc_classes[i][1])

	mean_tpr /= num_classes

	final_fpr = all_fpr
	final_tpr = mean_tpr

	return final_fpr, final_tpr