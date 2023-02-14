import pandas as pd
import numpy
import matplotlib.pyplot
import matplotlib.patches

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


    for i in range(1, 5):
        # load synthetic dataset 1
        data = pd.read_csv(f'data/synthetic-{i}.csv', header=None)
        features = data.values[:, :-1]
        labels = data.values[:, -1]
        # make sure labels are bools
        labels = labels.astype(bool)

        synthetic_tree = decision_tree.DecisionTree(features, labels, max_depth=3)

        # predict on the training data, measure accuracy
        predictions = [synthetic_tree.predict_label(feature) for feature in features]
        accuracy = numpy.mean(predictions == labels)
        print(f"Accuracy on synthetic-{i}: {accuracy * 100:.2f}%")


        # matplotlib.pyplot.subplot(2, 2, i)
        # plot the decision tree
        # fig = matplotlib.pyplot.figure()
        # ax = fig.add_subplot(111)
        ax = matplotlib.pyplot.subplot(2, 2, i)


        # set axis limits from data
        ax.set_xlim(numpy.min(features[:, 0]), numpy.max(features[:, 0]))
        ax.set_ylim(numpy.min(features[:, 1]), numpy.max(features[:, 1]))

        ax.set_aspect('equal')
        # ax scatter
        ax.scatter(features[:, 0], features[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(['red', 'blue']))
        # ax scatter legend
        red_patch = matplotlib.patches.Patch(color='red', label='False')
        blue_patch = matplotlib.patches.Patch(color='blue', label='True')
        if i == 4:
            ax.legend(handles=[red_patch, blue_patch])

        # render background
        x = numpy.linspace(numpy.min(features[:, 0]), numpy.max(features[:, 0]), 100)
        y = numpy.linspace(numpy.min(features[:, 1]), numpy.max(features[:, 1]), 100)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.array([synthetic_tree.predict_label([x, y]) for x, y in zip(numpy.ravel(X), numpy.ravel(Y))])
        Z = Z.reshape(X.shape)
        ax.contourf(X, Y, Z, cmap=matplotlib.colors.ListedColormap(['red', 'blue']), alpha=0.2)

    matplotlib.pyplot.show()

    # load pokemon dataset, strip headers
    features = pd.read_csv('data/pokemonStats.csv').values
    labels = pd.read_csv('data/pokemonLegendary.csv').values[:, 0]
    # make sure labels are bools
    labels = labels.astype(bool)

    pokemon_tree = decision_tree.DecisionTree(features, labels, max_depth=3)

    # predict on the training data, measure accuracy
    predictions = [pokemon_tree.predict_label(feature) for feature in features]
    accuracy = numpy.mean(predictions == labels)
    print(f"Accuracy on pokemon: {accuracy*100:.2f}%")

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)

    # set axis limits from data
    ax.set_xlim(numpy.min(features[:, 0]), numpy.max(features[:, 0]))
    ax.set_ylim(numpy.min(features[:, 1]), numpy.max(features[:, 1]))

    ax.set_aspect('equal')
    ax.set_title(f"Decision Tree for pokemon")
    # ax scatter
    ax.scatter(features[:, 0], features[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(['red', 'blue']))
    # ax scatter legend
    red_patch = matplotlib.patches.Patch(color='red', label='False')
    blue_patch = matplotlib.patches.Patch(color='blue', label='True')
    ax.legend(handles=[red_patch, blue_patch])

    # render background, this is a little different
    # x = numpy.linspace(numpy.min(features[:, 0]), numpy.max(features[:, 0]), 100)
    # y = numpy.linspace(numpy.min(features[:, 1]), numpy.max(features[:, 1]), 100)
    # X, Y = numpy.meshgrid(x, y)
    # Z = numpy.array([pokemon_tree.predict_label([x, y]) for x, y in zip(numpy.ravel(X), numpy.ravel(Y))])

    matplotlib.pyplot.show()




