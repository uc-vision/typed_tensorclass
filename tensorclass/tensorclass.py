from dataclasses import InitVar, asdict, dataclass, field, fields, replace
import torch

from torch import Tensor
from jaxtyping.array_types import _NamedVariadicDim

from jaxtyping import Float, Int, jaxtyped


def add_batch_dim(t, name='N', broadcast=True):
  if t.index_variadic is not None:
    if t.index_variadic != 0 or t.dims[0].name != name:
      raise ValueError(f"Index variadic must be first dimension with name {name}")
    
  else:
    t.dims=(_NamedVariadicDim(name, broadcast), *t.dims)
    t.index_variadic = 0

  return t





@dataclass(kw_only=True, repr=False)
class TensorClass():
  # Broadcast prefixes shapes together
  broadcast:  InitVar[bool] = False   


  def check_shapes(self, broadcast=False, memo=None, variadic_memo=None, variadic_broadcast_memo=None):
    memo = {} if memo is None else memo
    variadic_memo = {} if variadic_memo is None else variadic_memo
    variadic_broadcast_memo = {} if variadic_broadcast_memo is None else variadic_broadcast_memo

    for f in fields(self):
      value = getattr(self, f.name)
      batch_size = variadic_memo.get('N', None) if broadcast is False else variadic_broadcast_memo.get('N', None)
      
      if isinstance(value, torch.Tensor):
        t = add_batch_dim(f.type, broadcast=broadcast)

        if not t._check_shape(value, single_memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo):
          if batch_size is not None:
            batch = f"batch {tuple(*batch_size)}, "
          else:
            batch = ""

          class_name = self.__class__.__name__
          raise TypeError(f"{class_name}.{f.name}: bad shape {value.shape}, for {batch}shape ({t.dim_str}), broadcast={broadcast}")
        
      elif isinstance(value, TensorClass):
        value.check_shapes(broadcast=broadcast, memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo)

  def __post_init__(self, broadcast):
    self.check_shapes(broadcast=broadcast)





@dataclass
class T(TensorClass):
  a: Float[Tensor, "3"]
  b: Int[Tensor, "3"]
  c: str


@dataclass
class F(TensorClass):
  a: T
  b: Int[Tensor, "5"]
  c: str



if __name__ == "__main__":


  t = T(a=torch.randn(5, 1, 3), b=torch.randn(5, 1, 3).to(torch.long), c = "hello", broadcast=True)
  f = F(a=t, b=torch.randn(5, 7, 5).to(torch.long), c = "hello")

  # print(test_foo(t.a, t.b))
