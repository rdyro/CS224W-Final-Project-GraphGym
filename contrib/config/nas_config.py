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

    cfg.nas.node_0_1_act = "tanh"
    cfg.nas.node_0_2_act = "tanh"
    cfg.nas.node_0_3_act = "tanh"
    cfg.nas.node_1_2_act = "tanh"
    cfg.nas.node_1_3_act = "tanh"
    cfg.nas.node_2_3_act = "tanh"


register_config("nas", set_cfg_nas)
