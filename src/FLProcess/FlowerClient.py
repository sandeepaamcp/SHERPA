import flwr as fl

from src.NN.MdlTraining import train, test, test_multi_label, train_multi_label
from src.NN.NNConfig import get_nn
from src.NN.NNUtil import get_parameters, set_parameters
# from src.dataset.dataLoaderFactory import get_train_val_test_loaders, ClientConfig


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader):
        self.net = net
        self.trainloader = train_loader
        self.valloader = val_loader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        # train_multi_label(self.net, self.trainloader, epochs=1)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        # loss, accuracy = test_multi_label(self.net, self.valloader)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# def client_fn(cid) -> FlowerClient:
#     """Create a Flower client representing a single organization."""
#     # Load model
#     net = get_nn()
#     # trainloaders, valloaders, _ = get_train_val_test_loaders()
#     # Note: each client gets a different trainloader/valloader, so each client
#     # will train and evaluate on their own unique data
#
#     cid = config.cid
#     trainloaders = config.get_trainloader()
#     valloaders = config.get_valloader()
#
#     trainloader = trainloaders[int(cid)]
#     valloader = valloaders[int(cid)]
#
#     # Create a  single Flower client representing a single organization
#     return FlowerClient(net, trainloader, valloader)