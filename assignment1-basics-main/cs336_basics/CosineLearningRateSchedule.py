import math
def CosineLearningRateSchedule(t, alphamax, alphamin, Tw, Tc):

    if t<Tw:
        alpha = t/Tw * alphamax
    elif Tw<=t<=Tc:
        alpha =0.5*(1 + math.cos((t - Tw)/(Tc - Tw)*math.pi))*(alphamax - alphamin) + alphamin
    else:
        alpha = alphamin
    return alpha
