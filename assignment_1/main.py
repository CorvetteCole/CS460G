import pandas as pd
import numpy as np
import decision_tree

"""
all data files are stored in data/
below are the filenames and their first two rows
- pokemonLegendary.csv:
Legendary
False
- pokemonStats.csv:
Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Type 1_Bug,Type 1_Dark,Type 1_Dragon,Type 1_Electric,Type 1_Fairy,Type 1_Fighting,Type 1_Fire,Type 1_Flying,Type 1_Ghost,Type 1_Grass,Type 1_Ground,Type 1_Ice,Type 1_Normal,Type 1_Poison,Type 1_Psychic,Type 1_Rock,Type 1_Steel,Type 1_Water,Type 2_Bug,Type 2_Dark,Type 2_Dragon,Type 2_Electric,Type 2_Fairy,Type 2_Fighting,Type 2_Fire,Type 2_Flying,Type 2_Ghost,Type 2_Grass,Type 2_Ground,Type 2_Ice,Type 2_Normal,Type 2_Poison,Type 2_Psychic,Type 2_Rock,Type 2_Steel,Type 2_Water
318,45,49,49,65,65,45,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
- synthetic-[1-4].csv: No headers, 3 columns of floats:
10.58,-0.055609,1
10.813,0.77631,1
"""

if __name__ == "__main__":
    # load synthetic dataset 1
    data = pd.read_csv('data/synthetic-1.csv', header=None)
    x_values = data.values[:, :-1]
    y_values = data.values[:, -1]
    # make sure y values are bools
    y_values = y_values.astype(bool)

    # dt = DecisionTree(x_values, y_values, max_depth=3)
    # dt.fit(x_values, y_values)
    #
    # # Test the decision tree on the synthetic dataset
    # y_predictions = dt.predict(x_values)
    #
    # # Calculate the accuracy
    # error = np.mean(y_values != y_predictions)
    #
    # print(f"Error: {error:.3f}")
