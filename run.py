
from src.models.GCN_Trainer import *
from src.data.make_dataset import *
from src.features.build_features import *
from src.features.build_test_features import *

import sys


def main(targets):

    data, edges = make_data()

    if targets and targets[0] == 'test':

        print("Building test dataset")

        features, adj, labels = build_test_features(data, edges)

    else:

        features, adj, labels = build_features(data, edges)

    trainer = GCN_Trainer(features, adj, labels)

    trainer.complete_train()



if __name__ == '__main__':

    targets = sys.argv[1:]

    main(targets)
