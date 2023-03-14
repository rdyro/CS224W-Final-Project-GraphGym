from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_nas(cfg):
    r"""
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    """

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.nas = CN()
    cfg.nas.node0 = "GCN"
    cfg.nas.node1 = "GCN"
    cfg.nas.node2 = "GCN"
    cfg.nas.node3 = "GCN"

    cfg.nas.node01_act = "tanh"
    cfg.nas.node02_act = "tanh"
    cfg.nas.node03_act = "tanh"
    cfg.nas.node12_act = "tanh"
    cfg.nas.node13_act = "tanh"
    cfg.nas.node23_act = "tanh"


register_config("nas", set_cfg_nas)
