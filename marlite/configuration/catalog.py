"""Default components; optional environment/model families are loaded on demand."""

from torch import nn
import torch

from .registry import Registry
from marlite.algorithm.components import HyperNetwork, MaskChannels, Permute, sequential


def _dtype(name):
    allowed = {name: getattr(torch, name) for name in ("float16", "float32", "float64", "bfloat16")}
    if name not in allowed:
        raise ValueError(f"Unknown floating dtype {name!r}")
    return allowed[name]


def default_registry():
    registry = Registry()
    for name in (
        "Identity",
        "Linear",
        "Bilinear",
        "Flatten",
        "Unflatten",
        "ReLU",
        "GELU",
        "ELU",
        "SiLU",
        "Tanh",
        "Sigmoid",
        "Softplus",
        "Softmax",
        "LayerNorm",
        "GroupNorm",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "Upsample",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveMaxPool2d",
        "MaxPool1d",
        "MaxPool2d",
        "AvgPool2d",
        "MultiheadAttention",
        "Dropout",
    ):
        registry.register(name, getattr(nn, name))
    for name, factory in (
        ("Sequential", sequential),
        ("MaskChannels", MaskChannels),
        ("Permute", Permute),
        ("HyperNetwork", HyperNetwork),
        ("DType", _dtype),
    ):
        registry.register(name, factory)
    families = {
        "marlite.algorithm.temporal": [
            "FrameEncoder",
            "RecurrentGRU",
            "CausalAttention",
            "WindowEncoder",
        ],
        "marlite.algorithm.sequence_agent_group": ["SequenceAgentGroup"],
        "marlite.algorithm.sequence_value": ["SequenceValue", "TeamSequenceEncoder"],
        "marlite.algorithm.sequence_mixers": [
            "MonotonicMixing",
            "DuplexTransformation",
            "JointAdvantageWeights",
            "SequenceQMixer",
            "SequenceDuplexMixer",
            "CounterfactualJoint",
            "SequenceCounterfactualMixer",
        ],
        "marlite.communication.fusion": [
            "MaskedMeanPool",
            "MaskedMaxPool",
            "AttentionPool",
            "SetCommunication",
            "GaussianConsensus",
        ],
        "marlite.communication.graph": [
            "GraphCommunication",
            "DenseGraphModel",
            "GeometricGraphModel",
        ],
        "marlite.communication.group_set": ["GroupSetCommunication", "QuantizedCommunication"],
        "marlite.communication.context": [
            "AllTogether",
            "FixedGraph",
            "FixedGroups",
            "StateBuilderContext",
        ],
        "marlite.algorithm.model.group_communication": ["ScalarMessageCodec", "GroupSetFusion"],
        "marlite.algorithm.model.g2anet_attention": ["G2ANetAttention"],
        "marlite.algorithm.model.gnn": [
            "GCNModel",
            "GATModel",
            "SingleLayerGCNModel",
            "SingleLayerGATModel",
        ],
        "marlite.algorithm.model.matrix_gnn": ["MatrixGCNModel"],
        "marlite.algorithm.model.channel_selector": ["ChannelSelector"],
        "marlite.algorithm.model.resnet": [
            "ResidualMLP",
            "ResAttMaskedProbEnc",
            "ResAttMaskedStateEnc",
            "ResAttStateEnc",
            "ResAttObsEnc",
            "ResAttSeqEnc",
            "ResAttEnc",
            "SimpleResAttMaskedStateEnc",
            "SimpleResAttStateEnc",
            "SimpleResAttObsEnc",
            "SimpleResAttSeqEnc",
            "SimpleResAttEnc",
        ],
        "marlite.algorithm.model.decoder": ["NearestDecoder"],
        "marlite.algorithm.auxiliary.query_prediction": ["QueryPredictionHead"],
        "marlite.learning.value": ["QMIXObjective", "QPLEXObjective", "QTRANObjective"],
        "marlite.learning.auxiliary": [
            "ReconstructionObjective",
            "ObservationTargets",
            "ConstructorTargets",
        ],
        "marlite.learning.distillation": [
            "CompositeObjective",
            "FrameQueryTargets",
            "RelationQueryObjective",
        ],
        "marlite.util.loss_func": [
            "PITLoss",
            "PointSetMSELoss",
            "ChamferDistanceLoss",
            "InfoNCELoss",
        ],
        "marlite.util.loss_mixer": ["WeightedSumLoss", "EMAGradNormLoss", "PCGradLoss"],
        "marlite.util.communication_training": ["StateThresholdQueries"],
        "marlite.util.complementary_targets": ["ComplementaryAttackQueries"],
        "marlite.algorithm.group_builder.fixed_group_builder": ["FixedGroupBuilder"],
        "marlite.algorithm.group_builder.magent_group_builder": [
            "MAgentLabelPropagationGroupBuilder",
            "MAgentVecLPGroupBuilder",
            "MagentKMeansGroupBuilder",
            "MagentVecKMeansGroupBuilder",
        ],
        "marlite.algorithm.group_builder.partial_group_builder": [
            "PartialGroupMAgentBuilder",
            "PartialGroupVectorStateBuilder",
        ],
        "marlite.algorithm.graph_builder.fixed_edge_graph_builder": ["FixedEdgeGraphBuilder"],
        "marlite.algorithm.graph_builder.fullconn_graph_builder": ["FullConnGraphBuilder"],
        "marlite.algorithm.graph_builder.magent_graph_builder": [
            "MAgentGraphBuilder",
            "MAgentVecStateGraphBuilder",
        ],
        "marlite.algorithm.graph_builder.partial_graph_builder": [
            "PartialGraphMAgentBuilder",
            "PartialGraphVectorStateBuilder",
        ],
        "marlite.algorithm.graph_builder.random_group_graph_builder": ["RandomGroupGraphBuilder"],
        "marlite.util.self_supervised_data_constructor.magent_group_window_constructor": [
            "MagentGroupWindowConstructor"
        ],
        "marlite.util.self_supervised_data_constructor.magent_group_features_constructor": [
            "MagentGroupFeaturesConstructor"
        ],
        "marlite.util.self_supervised_data_constructor.magent_vec_state_group_features_constructor": [
            "MagentVecStateGroupFeaturesConstructor"
        ],
        "marlite.util.self_supervised_data_constructor.magent_obs_data_constructor": [
            "MagentVecObsDataConstructor",
            "MagentImageObsDataConstructor",
        ],
        "marlite.util.self_supervised_data_constructor.sumo_obs_data_constructor": [
            "SumoObsDataConstructor"
        ],
        "marlite.environment.smac_wrapper": ["SMACWrapper"],
        "marlite.environment.sumo_wrapper": ["SUMOWrapper"],
    }
    for module, names in families.items():
        for name in names:
            registry.register(name, (module, name))
    for name in ("Adam", "AdamW", "SGD", "RMSprop"):
        registry.register(name, getattr(torch.optim, name))
    return registry
