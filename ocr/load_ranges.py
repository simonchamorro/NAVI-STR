import os
import random


def interpolate(min_num, max_num):
    numbers = []
    num = min_num
    while num <= max_num:
        numbers.append(num)
        num += 2
    return numbers

def find_range(num, street, ranges):
    num_range = None
    for segment in ranges:
        if street == segment['street'] and num >= segment['min'] \
        and num <= segment['max'] and (segment['max'] - num) % 2 == 0:
            num_range = segment['range']
    return num_range

def get_random(num, street, ranges, qty):
    random_range = find_range(num, street, ranges)
    random.shuffle(random_range)
    if qty <= len(random_range):
        return random_range[:qty]
    else:
        return random_range

def get_sequence(num, street, ranges, qty):
    num_range = find_range(num, street, ranges)
    segment1 = num_range[:num_range.index(num)]
    segment2 = num_range[num_range.index(num) + 1:]
    if len(segment1) >= len(segment2):
        qty = len(segment1) if qty > len(segment1) else qty
        sequence = random.sample(segment1, qty)
        sequence.sort()
    elif len(segment2) > len(segment1):
        qty = len(segment1) if qty > len(segment2) else qty
        sequence = random.sample(segment2, qty)
        sequence.sort(reverse=True)
    return sequence

def load_ranges(filename):
    ranges = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[0] == '#':
                continue
            line = line.replace('\n', '').split('\t')
            if not line[4] == 'None':
                ranges.append({'street':line[0], 'side':line[1], 'from':line[2], 'to':line[3],\
                         'min':int(line[4].split('-')[0]), 'max':int(line[4].split('-')[1]),        \
                         'range':interpolate(int(line[4].split('-')[0]), int(line[4].split('-')[1]))})
    return ranges

if __name__ == '__main__':
    ranges = load_ranges('./data/address_ranges.txt')
    random_nums = get_random(6585, 'saint-laurent', ranges, 5)
    sequence_nums = get_sequence(6600, 'saint-urbain', ranges, 5)
    sequence_nums1 = get_sequence(43, 'jean-talon', ranges, 5)
    import pdb; pdb.set_trace()
