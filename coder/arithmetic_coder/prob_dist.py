import numpy as np
import functools


# define the cache decorator
cache = functools.lru_cache(maxsize=None)

class ProbabilityDist:
    """
    Wrapper around a probability dict
    """

    def __init__(self, prob_dict=None):
        self._validate_prob_dist(prob_dict)

        # NOTE: We use the fact that since python 3.6, dictionaries in python are
        # also OrderedDicts. https://realpython.com/python-ordereddict/
        self.prob_dict = prob_dict

    def __repr__(self):
        return f"ProbabilityDist({self.prob_dict.__repr__()}"

    @property
    def size(self):
        return len(self.prob_dict)

    @property
    def alphabet(self):
        return list(self.prob_dict)

    @property
    def prob_list(self):
        return [self.prob_dict[s] for s in self.alphabet]

    @classmethod
    def get_sorted_prob_dist(cls, prob_dict, descending=False):
        """
        Returns ProbabilityDist class object with sorted probabilities.
        By default, returns Probabilities in increasing order (descending=False), i.e.,
        p1 <= p2 <= .... <= pn (python-default)
        """
        return cls(dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=descending)))

    @classmethod
    def normalize_prob_dict(cls, prob_dict):
        """
        normalizes dict -> dict_norm so that the sum of values is 1
        wraps dict_norm as a ProbabilityDist
        """
        sum_p = sum(prob_dict.values())
        return cls({a: b / sum_p for a, b in prob_dict.items()})

    @property
    @cache
    def cumulative_prob_dict(self):
        """return a list of sum of probabilities of symbols preceeding symbol"""
        cum_prob_dict = {}
        _sum = 0
        for a, p in self.prob_dict.items():
            cum_prob_dict[a] = _sum
            _sum += p
        return cum_prob_dict

    @property
    @cache
    def entropy(self):
        entropy = 0
        for _, prob in self.prob_dict.items():
            entropy += -prob * np.log2(prob)
        return entropy

    def probability(self, symbol):
        return self.prob_dict[symbol]

    @cache
    def neg_log_probability(self, symbol):
        return -np.log2(self.probability(symbol))

    @staticmethod
    def _validate_prob_dist(prob_dict):
        """
        checks if each value of the prob dist is non-negative,
        and the dist sums to 1
        """

        sum_of_probs = 0
        for _, prob in prob_dict.items():
            assert prob >= 1e-9, "probabilities negative or too small cause stability issues"
            sum_of_probs += prob

        # FIXME: check if this needs a tolerance range
        if abs(sum_of_probs - 1.0) > 1e-8:
            raise ValueError("probabilities do not sum to 1")

class Frequencies:
    """
    Wrapper around a frequency dict
    NOTE: Frequencies is a typical way to represent probability distributions using integers
    """

    def __init__(self, freq_dict=None):

        # NOTE: We use the fact that since python 3.6, dictionaries in python are
        # also OrderedDicts. https://realpython.com/python-ordereddict/
        self.freq_dict = freq_dict

    def __repr__(self):
        return f"Frequencies({self.freq_dict.__repr__()}"

    @property
    def size(self):
        return len(self.freq_dict)

    @property
    def alphabet(self):
        return list(self.freq_dict)

    @property
    def freq_list(self):
        return [self.freq_dict[s] for s in self.alphabet]

    @property
    def total_freq(self) -> int:
        """returns the sum of all the frequencies"""
        return np.sum(self.freq_list)

    @property
    def cumulative_freq_dict(self) -> dict:
        """return a list of sum of probabilities of symbols preceeding symbol
        for example: freq_dict = {A: 7,B: 1,C: 3}
        cumulative_freq_dict = {A: 0, B: 7, C: 8}

        """
        cum_freq_dict = {}
        _sum = 0
        for a, p in self.freq_dict.items():
            cum_freq_dict[a] = _sum
            _sum += p
        return cum_freq_dict

    def frequency(self, symbol):
        return self.freq_dict[symbol]

    def get_prob_dist(self) -> ProbabilityDist:
        """_summary_

        Returns:
            _type_: _description_
        """
        prob_dict = {}
        for s, f in self.freq_dict.items():
            prob_dict[s] = f / self.total_freq
        return ProbabilityDist(prob_dict)