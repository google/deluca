from abc import abstractmethod
from jax import Array
import jax.numpy as jnp
import flax.struct

# TODO: Fix normalization.
class Normalizer:
    @abstractmethod
    def compute(self, a: Array) -> None:
        """Computes the necessary parameters from the input to normalize and denormalize future inputs."""

    @abstractmethod
    def normalize(self, a: Array) -> Array:
        """Normalizes the input."""

    @abstractmethod
    def denormalize(self, a: Array) -> Array:
        """Denormalizes the input."""

@flax.struct.dataclass
class NormalizerSet:
    obs: Normalizer
    action: Normalizer
    output: Normalizer

class WithoutNormalization(Normalizer):
    def compute(self, a: Array) -> None:
        pass

    def normalize(self, a: Array) -> Array:
        return a
    
    def denormalize(self, a: Array) -> Array:
        return a
    
class DefaultNormalizer(Normalizer):
    def compute(self, a: Array) -> None:
        flat_a = a.reshape(-1, a.shape[-1])
        self.mean = jnp.mean(flat_a, axis=0) 
        self.std = jnp.std(flat_a, axis=0)
    
    def normalize(self, a: Array) -> Array:
        return (a - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, a: Array) -> Array:
        return a * (self.std + 1e-8) + self.mean

NoNormalization = lambda: NormalizerSet(
    obs=WithoutNormalization(),
    action=WithoutNormalization(),
    output=WithoutNormalization(),
)

DefaultNormalizers = lambda: NormalizerSet(
    obs=DefaultNormalizer(),
    action=DefaultNormalizer(),
    output=DefaultNormalizer(),
)

