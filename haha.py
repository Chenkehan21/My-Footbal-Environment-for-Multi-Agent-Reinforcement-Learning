import math


def success_rate2(court_height, gate_width):
    gate_range = list(range(
                int(court_height / 2) - int(gate_width / 2), 
                int(court_height / 2) + math.ceil(gate_width / 2)))
    gate_center = []
    if len(gate_range) % 2 == 0:
        gate_center += [gate_range[len(gate_range) // 2 - 1], gate_range[len(gate_range) // 2]]
    else:
        gate_center.append(gate_range[len(gate_range) // 2])
    
    step1 = int((len(gate_range) - len(gate_center)) // 4)
    step2 = int((len(gate_range) - len(gate_center) - step1 * 2) / 2)
    range1 = gate_range[:step1] + gate_range[-step1:]
    range2 = gate_range[step1: step2 + step1] + gate_range[-step2 - step1 : -step1]
    success_rate = dict()

    for pos in gate_center:
        success_rate[pos] = 0.9

    for pos in range1:
        success_rate[pos] = 0.3
    
    for pos in range2:
        success_rate[pos] = 0.6

    print(success_rate)
    

success_rate2(20, 6)