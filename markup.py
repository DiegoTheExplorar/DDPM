import os
import sys
import random
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
