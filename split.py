"""Train/validation/test split that respects time ordering."""

def time_split(data, train=0.6, val=0.2):
    n = len(data)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return data[:t1], data[t1:t2], data[t2:]
