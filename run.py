
from src.models.GCN_Trainer import *
from src.data.make_dataset import *
from src.features.build_features import *

if __name__ == '__main__':


    data, edges = make_data()

    features, adj, labels = build_features(data, edges)

    trainer = GCN_Trainer(features, adj, labels)

    trainer.complete_train()
