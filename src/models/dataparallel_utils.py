"""
Enhanced DataParallel implementation for FlexibleCNN to ensure attribute access compatibility
"""

import torch
from torch.nn.parallel import DataParallel


class FlexibleDataParallel(DataParallel):
    """
    Custom DataParallel class that correctly handles attribute access
    by forwarding attribute lookups to the module being parallelized.
    
    This resolves issues with accessing attributes like 'pool_size' when using DataParallel.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
