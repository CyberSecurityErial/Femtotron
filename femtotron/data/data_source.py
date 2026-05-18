# dataset.py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    加载 preprocess() 输出。根据 meta 自动决定字段。
    
    Trainer 完全不知道 task 类型，看见的是统一的 (input_ids, labels[, ...]).
    """
    
    def __init__(self, path: str, ignore_index: int = -100, expect_task: str | None = None, mmap: bool = True):
        data = torch.load(path, weights_only=False, mmap=mmap)
        self.meta = data["meta"]
        self.ignore_index = ignore_index
        
        # 类型自证:防止 SFT 数据被当成 pretrain 用
        if expect_task is not None and self.meta["task"] != expect_task:
            raise ValueError(
                f"Loaded {path} (task={self.meta['task']}) "
                f"but expected task={expect_task}"
            )
        
        self.input_ids    = data["input_ids"]
        self.loss_mask    = data.get("loss_mask")       # SFT 才有
        self.position_ids = data.get("position_ids")    # 可能 None
        self.seqlens      = data.get("seqlens")          # 可能 None
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        
        # labels 运行时构造:bool mask 决定哪些位置算 loss
        if self.loss_mask is not None:
            mask = self.loss_mask[idx]
            labels = torch.where(mask, ids, self.ignore_index)
        else:
            labels = ids.clone()   # 预训练:全位置都算
        
        item = {"input_ids": ids, "labels": labels}
        if self.position_ids is not None:
            item["position_ids"] = self.position_ids[idx]
        if self.seqlens is not None:
            item["seqlens"] = self.seqlens[idx]
        return item

class PackedDataset(Dataset):
    """定长 token 序列的 dataset。
    
    输入是 preprocess.py 产出的 .pt 文件，shape [N, seq_len]。
    """
    
    def __init__(self, path: str, mmap: bool = True):
        # mmap=True 让多个 worker / 多个 rank 共享同一份内存映射
        self.data: Tensor = torch.load(path, mmap=mmap)
        assert self.data.dim() == 2, f"expected 2D tensor, got {self.data.shape}"
        assert self.data.dtype == torch.long, f"expected long, got {self.data.dtype}"
    
    @property
    def seq_len(self) -> int:
        return self.data.shape[1]
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]   # [seq_len], long