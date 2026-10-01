from torch.optim import Optimizer

class Scaffold_Optimizer(Optimizer):
    def __init__(self, parameter, learning_rate):
        default = {'lr': learning_rate}
        super().__init__(parameter, default)

    def step(self, intermediate_server_model_control, user_model_control, closure = None):
        loss = None
        if closure is not None:
            loss = closure
        
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], intermediate_server_model_control.values(), user_model_control.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']
        
        return loss