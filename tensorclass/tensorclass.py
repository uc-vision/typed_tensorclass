from dataclasses import InitVar, asdict, dataclass, field, fields, replace
from numbers import Number
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
  batched:  InitVar[bool] = True


  def check_shapes(self, broadcast=False, batched=True, memo=None, variadic_memo=None, variadic_broadcast_memo=None):
    memo = {} if memo is None else memo
    variadic_memo = {} if variadic_memo is None else variadic_memo
    variadic_broadcast_memo = {} if variadic_broadcast_memo is None else variadic_broadcast_memo

    self.batch_size = ()
    self.fields = {}


    for f in fields(self):
      value = getattr(self, f.name)    
      if isinstance(value, torch.Tensor):
        t = add_batch_dim(f.type, broadcast=broadcast) if batched else f.type

        if not t._check_shape(value, single_memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo):

          if self.batch_size is not None:
            batch = f"batch {tuple(*self.batch_size)}, "

          class_name = self.__class__.__name__
          raise TypeError(f"{class_name}.{f.name}: bad shape {value.shape}, for {batch}shape ({t.dim_str}), broadcast={broadcast}")
        
      elif isinstance(value, TensorClass):
        value.check_shapes(broadcast=broadcast, memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo)

      self.batch_size = variadic_memo.get('N', None) if broadcast is False else variadic_broadcast_memo.get('N', None)


  def __iter__(self):
    fs = fields(self)
    for f in fs:
      yield (f.name, getattr(self, f.name))

  def map(self, func):
    def f(t):
      if isinstance(t, torch.Tensor):
        return func(t)
      elif isinstance(t, TensorClass):
        return t.map(func)
      else:
        return t

    d = {k:f(t) for k, t in iter(self)}
    return self.__class__(**d)

  def __getitem__(self, slice):
    return self.map(lambda t: t[slice])

  def to(self, device):
    return self.map(lambda t: t.to(device))

  def expand(self, shape):
    return self.map(lambda t: t.expand(shape))

  @classmethod
  def empty(cls:type, shape=(), device='cpu', **kwargs):
    if isinstance(shape, Number):
      shape = (shape,)
    
    def make_tensor(k, info):
      if k in kwargs:
        return kwargs[k]

      if info is None:
        raise RuntimeError(f"{k}: has no argument or tensor annotation")

      if info.dtype is None:
        raise RuntimeError(f"{k}: has no dtype annotation")
      
      field_shape = (d.size for d in info.shape.dims)
      return torch.empty( tuple( (*shape, *field_shape) ), 
        dtype=info.dtype.dtype, device=device)


    return cls(**{f.name:make_tensor(f.name, self.(f.type)) 
      for f in fields(cls)}) 




  def unsqueeze(self, dim):
    assert dim <= len(self.shape), f"Cannot unsqueeze dim {dim} in shape {self.shape}"
    return self.map(lambda t: t.unsqueeze(dim))


  def __repr__(self):
    name= self.__class__.__name__
    return f"{name}({shape(asdict(self))})"



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
