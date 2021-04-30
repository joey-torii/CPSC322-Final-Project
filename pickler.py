import pickle
import os

from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation

stars_table = myutils.load_data("Stars.csv")
temperature = myutils.temp_bins(stars_table.get_column('Temperature'))
L = myutils.luminosity_bins(stars_table.get_column('L'))
R = myutils.get_radius(stars_table.get_column('R'))
a_m = myutils.get_magnitude(stars_table.get_column('A_M'))
color = myutils.categorize_colors(stars_table.get_column('Color'))
spectral_class = myutils.get_spectral_class(stars_table.get_column('Spectral_Class'))
star_type = stars_table.get_column('Type')

x_vals = [[temperature[i], str(L[i]), str(R[i]), str(a_m[i]), color[i], spectral_class[i]] for i in range(len(stars_table.data))]
y_vals = star_type

xtr, xts, ytr, yts = myevaluation.train_test_split(x_vals, y_vals)

my_tree = MyDecisionTreeClassifier()
my_tree.fit(xtr, ytr)

predicted = my_tree.predict(xts)
accuracy = myutils.compute_accuracy(predicted, yts)
print('My Decision Tree: Accuracy =', round(accuracy * 100, 3), 'Error Rate = ', round((1-accuracy) * 100, 3))

# pickle classifier
with open("decision_tree.p", "wb") as fout:
    pkl_obj = my_tree.tree
    pickle.dump(my_tree, fout)
