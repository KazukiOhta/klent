import jax

class KeyGenerator:
    def __init__(self, seed):
        self.key = jax.random.key(seed)

    def __call__(self):
        new_key, sub_key = jax.random.split(self.key)
        self.key = new_key
        return sub_key