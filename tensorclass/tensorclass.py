from dataclasses import InitVar, dataclass, fields, Field
from functools import cached_property
from numbers import Number
from typing import List
import torch
import numpy as np

from torch import Tensor
from jaxtyping.array_types import _NamedVariadicDim, _NamedDim, _FixedDim, AbstractArray, Float32, Int32

_batch_dim = '_B'

def _add_batch_dim(t, name=_batch_dim, broadcast=True):
  if t.index_variadic is not None:
    raise ValueError(f"Variadic dimension not allowed in TensorClass {t.dim_str}")
  else:
    return type(f"Batched{t.__name__})", (t,), dict(index_variadic=0, dims=(_NamedVariadicDim(name, broadcast), *t.dims)))



def array_variables(t:AbstractArray):
  vars = set()

  if t.index_variadic is not None:
    raise ValueError(f"Variadic dimension not allowed in TensorClass {t.dim_str}")

  for d in t.dims:
    if isinstance(d, _NamedDim):
      vars.add(d.name)
    elif isinstance(d, _FixedDim):
      return d.size
    else:
      raise ValueError(f"TensorClass does not support dimension {d}")
  
  return vars
    
def get_variables(t):
  if isinstance(t, AbstractArray):
    return array_variables(t)
  elif isinstance(t, TensorClass):
    return t.get_variables()

  return set()

def has_annotation(t):
  return issubclass(t, AbstractArray) or issubclass(t, TensorClass)


  
defaults = {
  'float32',
  'int32'
}

dtype_to_torch = dict(
  bfloat16=torch.bfloat16,
  float16=torch.float16,
  float32=torch.float32,
  float64=torch.float64,
  int8=torch.int8,
  int16=torch.int16,
  int32=torch.int32,
  int64=torch.int64,
  bool = torch.bool
)


def lookup_dtype(dtypes):
  if len(dtypes) > 1:
    for d in dtypes:
      if d in defaults:
        return dtype_to_torch[d]
  else:
    return dtype_to_torch[dtypes[0]]
  raise ValueError(f"Could not find default dtype in {dtypes}")    

def array_shape(t:AbstractArray, sizes):
  shape = []
  for d in t.dims:
    if isinstance(d, _NamedDim):
      if d.name in sizes:
        shape.append(sizes[d.name])
      else:
        raise ValueError(f"Missing size for variable {d.name}")
    elif isinstance(d, _FixedDim):
      shape.append(d.size)
    else:
      raise ValueError(f"TensorClass does not support dimension {d}")
  return tuple(shape)

def _broadcast(value, shape, batch_shape):
  if isinstance(value, torch.Tensor):
    target_shape = batch_shape + value.shape[-len(shape):]
    return value.broadcast_to(target_shape)
  elif isinstance(value, TensorClass):
    return value.broadcast_to(batch_shape)
  else:
    return value

def batch_from_memo(variadic_memo, variadic_broadcast_memo):
  if _batch_dim in variadic_memo:
    return tuple(variadic_memo[_batch_dim])
  elif _batch_dim in variadic_broadcast_memo:
    shapes = variadic_broadcast_memo[_batch_dim]
    return tuple(torch.broadcast_shapes(*shapes))
  

def arr_shape(t:AbstractArray):
  shape = []
  for d in t.dims:
    if isinstance(d, _NamedDim):
      shape.append(d.name)
    elif isinstance(d, _FixedDim):
      shape.append(d.size)
    else:
      raise ValueError(f"TensorClass does not support dimension {d}")
  return tuple(shape)

def arr_dims(t:AbstractArray):
  return len(t.dims)

def field_shape(f:Field, value:torch.Tensor):
  return tuple(value.shape[-arr_dims(f.type):])
    

    
@dataclass(kw_only=True, repr=False)
class TensorClass():
  # Broadcast prefixes shapes together
  broadcast:  InitVar[bool] = False
  convert_types:  InitVar[bool] = False


  def check_shapes(self, broadcast=False, convert_types=False, memo=None, variadic_memo=None, variadic_broadcast_memo=None):
    memo = {} if memo is None else memo
    variadic_memo = {} if variadic_memo is None else variadic_memo
    variadic_broadcast_memo = {} if variadic_broadcast_memo is None else variadic_broadcast_memo

    self.batch_shape = batch_from_memo(variadic_memo, variadic_broadcast_memo)
    self.fields = {}

    for f in fields(self):
      value = getattr(self, f.name)    

      if issubclass(f.type, AbstractArray):
        t = _add_batch_dim(f.type, broadcast=broadcast)

        if not isinstance(value, torch.Tensor):
          raise TypeError(f"{f.name} is not a Tensor")

        if not t._check_shape(value, single_memo=memo, 
              variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo):
          
          batch = ""
          if self.batch_shape is not None:
            batch = f"batch {tuple(self.batch_shape)}, "

          class_name = self.__class__.__name__
          raise TypeError(f"{class_name}.{f.name}: bad shape {value.shape}, for {batch}shape ({t.dim_str}), broadcast={broadcast}")
        
        dtype = str(value.dtype).split('.')[-1]
        if dtype not in f.type.dtypes:
          if convert_types:
            value = value.to(lookup_dtype(f.type.dtypes))
            setattr(self, f.name, value)
          else:
            raise TypeError(f"Type mismatch for {f.name}, expected one of {f.type.dtypes}, got {dtype}")

      elif issubclass(f.type, TensorClass):
        value.check_shapes(broadcast=broadcast, memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo)

      self.batch_shape = batch_from_memo(variadic_memo, variadic_broadcast_memo)



    if broadcast:  
      for f in fields(self):
        value = getattr(self, f.name)    
        if issubclass(f.type, AbstractArray):
          
          target_shape = self.batch_shape + field_shape(f, value)
          value = value.broadcast_to(target_shape)
        elif issubclass(f.type, TensorClass):
          value = value.broadcast_to(self.batch_shape)
        
        setattr(self, f.name, value)
      
  def broadcast_to(self, batch_shape):
    x = self.map_with_info(lambda _, arr, shape: _broadcast(arr, shape, batch_shape))
    return x


  @classmethod
  def shape_variables(cls):
    if not hasattr(cls, '_variables'):
      cls._variables = set()
      for f in fields(cls):
        cls._variables = get_variables(f.type)
    return cls._variables
  
  @property
  def shape(self):
    def g(f:Field):
      value = getattr(self, f.name)
      if issubclass(f.type, AbstractArray):
        return field_shape(f, value)
      elif issubclass(f.type, TensorClass):
        return value.shape
      else:
        return value
    return {f.name:g(f) for f in fields(self)}

  @cached_property
  def shape_info(self):
    def g(f:Field):
      value = getattr(self, f.name)
      if issubclass(f.type, AbstractArray):
        return (field_shape(f, value), value.dtype)
      elif issubclass(f.type, TensorClass):
        return value.shape_info
      else:
        return value
    return {f.name:g(f) for f in fields(self)}


  @classmethod
  def static_shape_info(cls):
    def g(f:Field):
      if issubclass(f.type, AbstractArray):
        return (arr_shape(f.type), lookup_dtype(f.type.dtypes))
      elif issubclass(f.type, TensorClass):
        return f.type.static_shape_info()
      else:
        return value
    return {f.name:g(f) for f in fields(cls)}
  

  @classmethod 
  def flat_size(cls):
    shapes = cls.static_shape_info()
    return sum([np.prod(s) for s, _ in shapes.values()])


  def items(self):
    fs = fields(self)
    for f in fs:
      yield (f.name, getattr(self, f.name))

      

  def map_with_info(self, func):
    def g(field):
      value = getattr(self, field.name)
      if issubclass(field.type, AbstractArray):
        shape = arr_shape(field.type)
        return func(field.name, value, shape=shape)
      elif isinstance(value, TensorClass):
        return value.map_with_info(func)
      else:
        return value

    d = {f.name:g(f) for f in fields(self)}
    return self.__class__(**d)



  def map(self, func):
    def g(field):
      value = getattr(self, field.name)
      if issubclass(field.type, AbstractArray):
        return func(value)
      elif isinstance(value, TensorClass):
        return value.map(func)
      else:
        return value

    d = {f.name:g(f) for f in fields(self)}
    return self.__class__(**d)

  def __getitem__(self, slice):
    return self.map(lambda t: t[slice])

  def to(self, device):
    return self.map(lambda t: t.to(device))

  def expand(self, shape):
    return self.map(lambda t: t.expand(shape))
  
  def reshape(self, *batch_shape):
    def g(name, value, shape):
      if isinstance(value, TensorClass):
        return t.reshape(batch_shape)
      if isinstance(value, torch.Tensor):
        return value.reshape(batch_shape + shape)
      else:
        return value
        
    return self.map_with_info(g)


  @classmethod
  def _build(cls:type, f, batch_shape=(), device=torch.device('cpu'), sizes=None, kwargs=None):
    sizes = {} if sizes is None else sizes
    
    if isinstance(batch_shape, Number):
      batch_shape = (batch_shape,)
    
    def make_tensor(t):
      if issubclass(t, AbstractArray):
        shape = batch_shape + array_shape(t, sizes)
        return f(shape, dtype=lookup_dtype(t.dtypes), device=device)
      elif issubclass(t, TensorClass):
        return t._build(f, batch_shape=batch_shape, device=device, sizes=sizes)

    tensors = {f.name: t  
        for f in fields(cls)
          if (t := make_tensor(f.type)) is not None}
    return cls(**tensors, **(kwargs or {}) )



  @classmethod
  def empty(cls:type, batch_shape=(), device=torch.device('cpu'), sizes=None, kwargs=None):
    return cls._build(torch.empty, batch_shape=batch_shape, device=device, sizes=sizes, kwargs=kwargs)

  @classmethod
  def zeros(cls:type, batch_shape=(), device=torch.device('cpu'), sizes=None, kwargs=None):
    return cls._build(torch.zeros, batch_shape=batch_shape, device=device, sizes=sizes, kwargs=kwargs)


  @cached_property
  def device(self):
    devices = [t.device for t in self.tensors()]
    if not all([d == devices[0] for d in devices]):
      raise ValueError("Inconsistent devices found")
    
    return devices[0]


  def unsqueeze(self, dim):
    assert dim <= len(self.batch_shape), f"Cannot unsqueeze dim {dim} in shape {self.shape}"
    return self.map(lambda t: t.unsqueeze(dim))
  
  def asdict(self) -> dict:
    def get_value(value):
      if isinstance(value, TensorClass):
        return value.asdict()
      else:
        return value
    return {k: get_value(value) for k, value in self.items()}
        
  def tensors(self) -> List[torch.Tensor]:
    x = []
    for value in self.asdict().values():
      if isinstance(value, TensorClass):
        x.extend(value.tensors())
      elif isinstance(value, torch.Tensor):
        x.append(value)

    return x

  def flat(self):
    return torch.concat([t.reshape(*self.batch_shape, -1) for t in self.tensors()], dim=-1)

  def __repr__(self):
    name= self.__class__.__name__

    info = ", ".join([f"{k}={v}" for k, v in self.shape_info.items()])
    return f"{name}({info}, batch_shape={self.batch_shape})"


  def __post_init__(self, broadcast,  convert_types):
    self.check_shapes(broadcast=broadcast,  convert_types=convert_types)




if __name__ == "__main__":

  @dataclass
  class T(TensorClass):
    a: Float32[Tensor, "K"]
    b: Int32[Tensor, "K"]
    c: str = "Hello"


  @dataclass
  class F(TensorClass):
    i: T
    j: Int32[Tensor, "5"]
    blah: Float32[Tensor, "5"] = torch.Tensor([1.0])



  x = T(a=torch.randn(1, 4, 3), b=torch.randn(4, 1, 3).to(torch.long), c = "hello", broadcast=True)
  # y = T(a=torch.randn(5, 1, 3), b=torch.randn(5, 1, 3).to(torch.long), c = "hello", broadcast=True)
  f = F(i=x, j=torch.randn(7, 1, 1, 5).to(torch.long), broadcast=True)

  print(f._shape_info())
  # print(test_foo(t.a, t.b))
