import lightning.pytorch as pl
import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp 
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


class SimpleDataModule(pl.LightningDataModule):

    def __init__(self,
                 ds_train: object,
                 ds_val:object =None,
                 ds_test:object =None,
                 batch_size: int = 1,
                 num_workers: int = mp.cpu_count(),
                 seed: int = 0, 
                 pin_memory: bool = False,
                 persistent_workers: bool = False,
                 weights: list = None,
                 balanced_epoch=False,
                 sampler_num_samples: int | None = None,
                ):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.ds_train = ds_train 
        self.ds_val = ds_val 
        self.ds_test = ds_test 

        self._n_classes = len(getattr(ds_train, "CLASS2IDX", {})) or 1

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed 
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.weights = weights
        
        self.balanced_epoch=balanced_epoch
        self.sampler_num_samples = sampler_num_samples

   

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.weights is not None:
          if self.balanced_epoch:
            num_samples = self.sampler_num_samples or self._n_classes * 1_000
            sampler = WeightedRandomSampler(
                self.weights, num_samples, generator=generator
            )
          else:
            sampler = WeightedRandomSampler(self.weights, len(self.weights), generator=generator) 
        else:
            sampler = RandomSampler(self.ds_train, replacement=False, generator=generator)
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, 
                          sampler=sampler, generator=generator, drop_last=False, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers if self.num_workers > 0 else False)


    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                                generator=generator, drop_last=False, pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers if self.num_workers > 0 else False)
        else:
            raise AssertionError("A validation set was not initialized.")


    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                            generator = generator, drop_last=False, pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers if self.num_workers > 0 else False)
        else:
            raise AssertionError("A test test set was not initialized.")