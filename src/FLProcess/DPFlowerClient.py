from src.FLProcess.FlowerClient import FlowerClient
from src.NN.MdlTraining import train_dp
from src.NN.NNUtil import get_parameters, set_parameters


class DPFlowerClient(FlowerClient):
    def __init__(self, net, train_loader, val_loader, dp_config):
        self.net = net
        self.trainloader = train_loader
        self.valloader = val_loader
        self.dp_config = dp_config

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_dp(self.net, self.trainloader, args=self.dp_config, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
