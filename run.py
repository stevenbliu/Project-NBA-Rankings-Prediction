
# from src.features.build_features import *
# from src.models.GCN import *

from src.main import *

if __name__ == '__main__':

    GCN_train()


    # stats = get_season_stats(2011)
    # rankings = get_season_ranks(2011)
    #
    # features, labels = get_features(2011)
    #
    # adj = get_adjacency(2011)
    #
    # #eventually
    # model = GCN(nfeat=features.shape[1],
    #             nhid=16,
    #             nclass=len(labels) + 1,
    #             dropout=0.5)
    #
    # optimizer = optim.Adam(model.parameters(),
    #                 lr=.01, weight_decay=5e-4)
    #
    # # Train model
    # t_total = time.time()
    # for epoch in range(100):
    #     train(epoch)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    #
    # # Testing
    # test()
