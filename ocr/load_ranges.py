import random


def interpolate(min_num, max_num):
    numbers = []
    num = min_num
    while num <= max_num:
        numbers.append(num)
        num += 2
    return numbers


def find_range(num, street, ranges):
    '''
    Find address ranges for a given segment street.

    Args:
        num (int): Targeted house number.
        street (str): Street segment.
        ranges (list): list of dict containing street segment information and
        address ranges. E.g. {'street': 'saint-laurent', 'side': 'E',
        'from': 'beaubien', 'to': 'saint-zotique', 'min': 6537, 'max': 6743,
        'range': [6537, 6539, 6541, 6543]}
    '''
    num_range = None
    for segment in ranges:
        if street == segment['street'] and num >= segment['min'] \
          and num <= segment['max'] and (segment['max'] - num) % 2 == 0:
            num_range = segment['range']
    if num_range is None:
        print(street)
        print(num)
    return num_range


def get_random(num, street, qty):
    '''
    Get random neighboor house number from an address in the street segment.
    Note that you can get the targeted house number.

    Args:
        num (int): Targeted house number.
        street (str): Street segment.
        qty (int): Number of neighboor house number to sample.

    Returns:
        random_range (list): List of neighboor house number.
    '''
    ranges = load_ranges('./data/address_ranges.txt')
    random_range = find_range(num, street, ranges)
    if qty <= len(random_range):
        random.shuffle(random_range)
        return random_range[:qty]
    else:
        random_range = random.choices(random_range, k=qty)
        return random_range


def get_sequence(num, street, qty):
    ranges = load_ranges('./data/address_ranges.txt')
    num_range = find_range(num, street, ranges)
    import pdb; pdb.set_trace()
    segment1 = num_range[:num_range.index(num)]
    segment2 = num_range[num_range.index(num) + 1:]
    if random.choice([True, False]):
        sequence = random.choices(segment1, k=qty)
        sequence.sort()
    else:
        sequence = random.choices(segment2, k=qty)
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
    random_nums = get_random(6585, 'saint-laurent', 5)
    sequence_nums = get_sequence(6600, 'saint-urbain', 5)
    sequence_nums1 = get_sequence(43, 'jean-talon', 5)
    import pdb; pdb.set_trace()
