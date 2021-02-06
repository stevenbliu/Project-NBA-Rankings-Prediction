
from src.models.GCN import *
from src.features.build_features import *
from src.data.make_dataset import *


def GCN_train():

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        print(output.shape, labels.shape)

        loss = nn.CrossEntropyLoss()
        loss_train = loss(output, labels.type(torch.LongTensor))
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()


        loss_val = loss(output, labels.type(torch.LongTensor))
        acc_val = accuracy(output, labels)

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


    def test():
        model.eval()
        output = model(features, adj)
        loss = nn.CrossEntropyLoss()
        loss_test = loss(output, labels.type(torch.LongTensor))
        acc_test = accuracy(output, labels)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))


    stats = get_season_stats(2011)
    rankings = get_season_ranks(2011)

    features, labels = get_features(2011)

    adj = get_adjacency(2011)

    #eventually
    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=len(labels) + 1,
                dropout=0.5)

    optimizer = optim.Adam(model.parameters(),
                    lr=.01, weight_decay=5e-4)

    # Train model
    t_total = time.time()
    for epoch in range(100):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
