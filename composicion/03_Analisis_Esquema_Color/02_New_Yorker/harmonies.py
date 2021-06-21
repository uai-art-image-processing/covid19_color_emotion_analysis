import numpy as np

# get color harmony from mainColors
def getHarmony(wheelColors):
    # Get the corresponding colors on the RGB wheel from array of RGB colors
    w = wheelColors
    harmonies = np.array([monochromatic(w), complementary(w), splitComplementary(w), triad(w), square(w), rectangular(w), analogous(w), False]) * 1
    
    # If other color combinations
    if sum(w) > 1 and sum(harmonies)==0:
        harmonies[-1] = 1
    return harmonies
    
# Harmonies    
def monochromatic(wheelColors):
    return sum(wheelColors)==1


def complementary(wheelColors):
    for curr in range(12):
        opp = (curr+6) % 12
        if wheelColors[curr]==1 and wheelColors[opp]==1:
            return True
    return False


def splitComplementary(wheelColors):
    for curr in range(12):
        opp_left = (curr+5) % 12
        opp_right = (curr+7) % 12
        if wheelColors[curr]==1 and wheelColors[opp_left]==1 and wheelColors[opp_right]==1:
            return True
    return False


def triad(wheelColors):
    for curr in range(12):
        left = (curr+4) % 12
        right = (curr+8) % 12
        if wheelColors[curr]==1 and wheelColors[left]==1 and wheelColors[right]==1:
            return True
    return False


def square(wheelColors):
    for curr in range(12):
        left = (curr+3) % 12
        right = (curr+9) % 12
        opp = (curr+6) % 12
        if wheelColors[curr]==1 and wheelColors[left]==1 and wheelColors[right]==1 and wheelColors[opp]==1:
            return True
    return False


def rectangular(wheelColors):
    for curr in range(6):
        for width in range(1,3):
            left = (curr+width) % 12
            right = (curr+6+width) % 12
            opp = (curr+6) % 12
            if wheelColors[curr]==1 and wheelColors[left]==1 and wheelColors[right]==1 and wheelColors[opp]==1:
                return True
    return False


def analogous(wheelColors):
    for curr in range(12):
        right = (curr+1) % 12
        if wheelColors[curr]==1 and wheelColors[right]==1:
            return True
    return False