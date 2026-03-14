import numpy as np
from .hrr import HRR, HRRArray
from typing import Callable
from numpy.typing import ArrayLike, DTypeLike

class CleanupMemory:
    """Stub class for generic cleanup memories.

    Attributes:
        n (int): Number of stored vectors (may change); may differ from reserved memory.
        d (int): Number of dimensions of stored vectors (may not be changed).
        dtype (dtype): Type of the vectors stored in cleanup memory.
        sim (Callable): Function that computes the similarity between two vectors in case they 
                        aren't HRRs. Default cosine similarity.
    """

    def __init__(self, n:int, d:int, *args, 
                 dtype:DTypeLike=np.float64, 
                 sim:Callable[[ArrayLike, ArrayLike], DTypeLike] = lambda u, v: (u@v)/(u@u * v@v),
                 **kwargs
                 ):
        """Initialize an n by d cleanup memory.

        Args:
            n (int): Number of vectors in the cleanup memory.
            d (int): Dimension of vectors to be stored in the memory (must be constant).
            dtype (DTypeLike): numpy dtype for the memory.
        
        Returns:
            None
        """

        self.dtype = np.dtype(dtype) # unused for now, may be necessary with FHRRs
        self.n = n
        self.d = d
        self.sim = sim

    def nearest(self, 
                probe:np.ndarray | HRR, 
                k:int=1
                ) -> tuple[HRRArray, np.ndarray]:
        """Returns k nearest neighbours of probe using probe's similarity measure or self.sim 
        function if unavailable (default 1), and similarities of neighbours, in descending order.

        Args:
            probe (np.ndarray | HRR): The vector whose neighbours you want to find.
            k (int): Number of neighbours to find.

        Returns:
            HRRArray: The top k HRRs in the search (greatest similarity first).
            ndarray: The similarities of the top k items
        """

        return HRRArray(np.zeros((k, self.d))), np.zeros(k)

    def insert(self,
               trace: np.ndarray | HRR
               ):
        """Adds a new HRR into the memory, expanding it if necessary.
        
        Args:
            trace (np.ndarray | HRR): the vector to add to memory.

        Returns:
            None
        """
        pass

    def __getitem__(self, probe:HRR, showsim:bool=False
                    ) -> HRR | tuple[HRR, DTypeLike]:
        """Fetches the nearest neighbour of the probe as a dictionary lookup.

        Args:
            probe (HRR): the HRR whose neighbour we're getting.
            showsim (bool): whether to return the similarity as well.

        Returns:
            HRR: the nearest neighbour of probe.
            self.dtype: the similarity of probe with the returned trace.
        """
        if showsim:
            return self.nearest(probe, 1)

        return self.nearest(probe, 1)[0][0]

class NaiveMemory(CleanupMemory):
    """Naive cleanup memory; searches by getting the k nearest neighbours with a probe and
    all items.

    Attributes:
        n (int): Number of stored vectors (may change); may differ from reserved memory.
        d (int): Number of dimensions of stored vectors (may not be changed).
        dtype (dtype): Type of the vectors stored in cleanup memory.
        sim (Callable): Function that computes the similarity between two vectors in case they 
                        aren't HRRs. Default cosine similarity.
    """

    def __init__(self, n:int, d:int, 
                 data: HRRArray | None = None,
                 dtype:DTypeLike=np.float64, 
                 sim:Callable[[ArrayLike, ArrayLike], DTypeLike] = lambda u, v: (u@v)/(u@u * v@v),
                 **kwargs
                 ):
        """Initialize an n by d cleanup memory.

        Args:
            n (int): Number of vectors in the cleanup memory.
            d (int): Dimension of vectors to be stored in the memory (must be constant).
            dtype (DTypeLike): numpy dtype for the memory.
        
        Returns:
            None
        """

        self.dtype = np.dtype(dtype) # unused for now, may be necessary with FHRRs
        self.n = 0
        self.d = d
        self.sim = sim
        internal_n = 2**int(np.ceil(np.log2(n)))
        self.memory = HRRArray(internal_n, d, data=np.zeros((internal_n, d), dtype=self.dtype))
        self.populate(data)

    def populate(self,
                 data: np.ndarray | HRRArray):
        """Adds several HRRs to the memory.

        Args:
            data (np.ndarray | HRRArray): HRRs to append.

        Returns:
            None
        """
    
        k = len(data)

        self._grow(k)

        if isinstance(data, HRRArray):
            self.memory.M[self.n:self.n+k, :] = data.M[:, :]
        else:
            self.memory.M[self.n:self.n+k, :] = data[:, :]

        self.n += k

    def _grow(self, k:int):
        """Doubles memory size if n + k > len(memory)

        Args:
            k (int): number of items to be added.

        Returns:
            None
        """

        if self.n + k > len(self.memory):
            internal_n = 2**int(np.ceil(np.log2(self.n + k)))
            newmem = np.zeros((internal_n, self.d), dtype=self.dtype)
            newmem[:self.n, :] = self.memory.M[:self.n, :]
            self.memory = HRRArray(internal_n, self.d, data=newmem)


    def nearest(self, 
                probe:np.ndarray | HRR, 
                k:int=1
                ) -> tuple[HRRArray, np.ndarray]:
        """Returns k nearest neighbours of probe using probe's similarity measure or self.sim 
        function if unavailable (default 1), and similarities of neighbours, in descending order.

        Args:
            probe (np.ndarray | HRR): The vector whose neighbours you want to find.
            k (int): Number of neighbours to find.

        Returns:
            HRRArray: The top k HRRs in the search (greatest similarity first).
        """

        sims = np.zeros(self.n)

        if isinstance(probe, HRR):
            sims[:] = (self.memory == probe)[:self.n]
        else:
            sims[:] = [self.sim(self.memory.M[i, :], probe) for i in range(self.n)]

        sorted_order = np.argpartition(sims[:self.n], -k)[::-1] # permutation that sorts the similarities
        top_items = sorted_order[:k]

        return HRRArray(k, self.d, data=self.memory[top_items]), sims[top_items]

    def insert(self,
               trace: np.ndarray | HRR
               ):
        """Adds a new HRR into the memory, expanding it if necessary.
        
        Args:
            trace (np.ndarray | HRR): the vector to add to memory.

        Returns:
            None
        """
        self._grow(1)

        if isinstance(trace, HRR):
            self.memory.M[self.n, :] = trace.v[:]
        else:
            self.memory.M[self.n, :] = trace[:]

        self.n += 1
    
    def free(self, k:int):
        """Frees k rows of the memory, without zeroing.

        Args:
            k (int): the number of rows to free.

        Returns:
            None
        """

        self.n -= k
