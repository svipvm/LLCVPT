from yacs.config import CfgNode as CN

# ------------------------------------------------------------------------------------------ #
# Config definition
# ------------------------------------------------------------------------------------------ #
_C = CN()

# ------------------------------------------------------------------------------------------ #
# Task
# ------------------------------------------------------------------------------------------ #
_C.TASK = CN()
_C.TASK.NAME = ""                       # the name of task (about record directory)
_C.TASK.VERSION = ""                    # the desciption of this task (for search)
_C.TASK.DEVICES = []                    # the task device: [0,..] or [] (gpu or cpu)
_C.TASK.IMG_SCALE = None                # the scale of task image (for super-resolution)
_C.TASK.IMG_CHANNELS = None             # the channel number of this task (gray:1, rgb:3)

# ------------------------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------------------------ #
_C.DATASETS = CN()
_C.DATASETS.TYPE = ""                   # the type of dataset (relate to a dataset class)
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = ""             # just a name about the train dataset
_C.DATASETS.TRAIN.H_DATASETS = ()       # the paths about hihg quality image
_C.DATASETS.TRAIN.L_DATASETS = ()       # the paths about low quality image (nonessential)
_C.DATASETS.TRAIN.PATCH_SIZE = None     # the size(patch, patch) of image inter to trainer
_C.DATASETS.TRAIN.BATCH_SIZE = None     # the size about a batch of dataloader
_C.DATASETS.TRAIN.NUM_WORKERS = None    # the thread number about load train sets
_C.DATASETS.TRAIN.SIGMA = None          # a single sigma value for attach image noising
_C.DATASETS.TRAIN.SIGMA_RANGE = []      # a sigma range for attach image noising
_C.DATASETS.VALID = CN()
_C.DATASETS.VALID.NAME = ""             # just a name about the valid dataset (test-sets during train)
_C.DATASETS.VALID.H_DATASETS = ()       # the paths about hihg quality image  
_C.DATASETS.VALID.L_DATASETS = ()       # the paths about low quality image (nonessential)
_C.DATASETS.VALID.SIGMA = None          # a single sigma value for attach image noising
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = ""              # just a name about the valid dataset (apply to the final model)
_C.DATASETS.TEST.H_DATASETS = ()        # the paths about hihg quality image  
_C.DATASETS.TEST.L_DATASETS = ()        # the paths about low quality image (nonessential)
_C.DATASETS.TEST.SIGMA = None           # a single sigma value for attach image noising

# ------------------------------------------------------------------------------------------ #
# Generation Model
# ------------------------------------------------------------------------------------------ #
_C.MODELG = CN()
_C.MODELG.TYPE = ""                     # the type of a generation model (relate to a model class)
_C.MODELG.IN_CHANNELS = None            # number of channels entering the model
_C.MODELG.MOD_CHANNELS = []             # number of channels related to number of layers
_C.MODELG.NUM_LAYERS = []               # number of layers related to channels
_C.MODELG.OUT_CHANNELS = None           # number of channers out the model
_C.MODELG.ACT_MODE = ""                 # the types of activation functions
_C.MODELG.UPSAMPLE_MODE = ""            # the type of upsample in the model
_C.MODELG.DOWNSAMPLE_MODE = ""          # the type of downsample in the model
_C.MODELG.PRETRAINED = ""               # A path of pretrained weight

# ------------------------------------------------------------------------------------------ #
# Loss function
# ------------------------------------------------------------------------------------------ #
_C.LOSS = CN()
_C.LOSS.TYPE = ""                       # the type of loss funtion (relate to a function class)

# ------------------------------------------------------------------------------------------ #
# Solver
# ------------------------------------------------------------------------------------------ #
_C.SOLVER = CN()
_C.SOLVER.TYPE = ""                     # the type of solver (relate to a optimizer class)
_C.SOLVER.NUM_EPOCHS = None             # the number of epoch during train (necessary)
_C.SOLVER.BASE_LR = None                # the learing rate in the solver (necessary)
_C.SOLVER.BETAS = []                    # the betas in the solver about Adam
_C.SOLVER.WEIGHT_DECAY = None           # the weight decay in the solver (necessary)
_C.SOLVER.MOMENTUM = None               # the betas in the solver about SGD

# ------------------------------------------------------------------------------------------ #
# Scheduler
# ------------------------------------------------------------------------------------------ #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ""                  # the type of scheduler (relate to a scheduler class)
_C.SCHEDULER.MILESTONES = []            # list of epoch indices, must be increasing
_C.SCHEDULER.GAMMA = None               # factor of learning rate decay

# ------------------------------------------------------------------------------------------ #
# Record
# ------------------------------------------------------------------------------------------ #
_C.RECORD = CN()
_C.RECORD.OUTPUT_DIR = ""               # the root directory of output task results
_C.RECORD.TIME_STAMP = ""               # automatically generated task timestamp
_C.RECORD.LOG_PERIOD = None             # iteration number per log during train (necessary)
_C.RECORD.TEST_PERIOD = None            # iteration number per test during train (necessary)
_C.RECORD.SAVE_PERIOD = None            # iteration number per save model during train (necessary)

# ------------------------------------------------------------------------------------------ #
# Option
# ------------------------------------------------------------------------------------------ #
_C.OPTION = CN()
_C.OPTION.SEED = None                   # random seeds in each random library

# ------------------------------------------------------------------------------------------ #
# Test
# ------------------------------------------------------------------------------------------ #
_C.TEST = CN()
_C.TEST.WEIGHT = ""                     # the weight file of test model
