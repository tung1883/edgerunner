
def fixed_fraction(capital, fraction=0.1):
    return capital * fraction

def kelly(win_rate, win_loss_ratio):
    return win_rate - (1 - win_rate) / win_loss_ratio

