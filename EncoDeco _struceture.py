from torch import nn
# 编码器
class encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return x


# 解码器
class decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    # init i 接受编码器传出的状态, 对自身初始化
    def init_state(self, encod_opt, *args):
        state = encod_opt
        return state
    def forward(self, x, state):
        return NotImplementedError

class enco_deco(nn.Module):
    def __init__(self, enco, deco) -> None:
        super().__init__()
        self.enco = enco
        self.deco = deco
    def forward(self, enco_x, deco_x,*args):
        enco_out = self.enco(enco_x)
        state = self.deco.init_state(enco_out)
        x, _ = self.deco(deco_x, state)
        return x