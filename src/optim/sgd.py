import torch.optim as optim


class SGD(optim.SGD):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr=lr)

    def step(self, closure=None, iteration=0):
        loss = super().step(closure)

        if iteration != 0:
            gradient_list = []
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        gradient_list.append(param.grad.data.clone())
            return gradient_list
        return loss
