
import random

from agent import Agent
from neuro.areas.encoder_area import EncoderArea
from neuro.hyper_params import HyperParameters
from neuro.container import Container
from neuro.network import Network
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.neural_zone import NeuralZone
from neuro.sdr_processor import SDRProcessor

space_size = 1000
value_size = 20
num_trials = 100
shift_coeff = 0.3
recognition_threshold = 0.8


def change_pattern(pattern, coefficient):
    num_neurons_to_be_replaced = int(pattern.value_size * coefficient)
    new_value = random.sample(pattern.value, pattern.value_size - num_neurons_to_be_replaced)
    while len(new_value) < pattern.value_size:
        val = random.choice(range(pattern.space_size))
        if val not in new_value:
            new_value.append(val)
    new_value.sort()
    return NeuralPattern(space_size=pattern.space_size, value=new_value)


def main():
    agent = Agent()

    random.seed(2)

    zone = NeuralZone(name='fake', agent=agent)
    area = EncoderArea.add(
        name='representations',
        agent=agent,
        zone=zone,
        output_space_size=space_size,
        output_norm=value_size,
        recognition_threshold=recognition_threshold
    )

    sdr = SDRProcessor(area)
    common_pattern = NeuralPattern(space_size=space_size, value_size=value_size)
    common_pattern.generate_random()

    patterns = []

    false_positives = 0
    false_negatives = 0
    for i in range(num_trials):
        random_pattern = NeuralPattern(space_size=space_size, value_size=value_size)
        random_pattern.generate_random()

        combined_pattern = SDRProcessor.make_combined_pattern(
            [random_pattern, common_pattern],
            [space_size, space_size]
        )

        out_pattern1, new = area.recognize_process_input(combined_pattern)

        if not new:
            false_positives += 1
        # print(f'{i}: {new} {out_pattern1}')

        shifted_pattern = change_pattern(combined_pattern, shift_coeff)
        out_pattern1, new = area.recognize_process_input(shifted_pattern)
        if new:
            false_negatives += 1

        # patterns.append(out_pattern1)
        #
        # if len(patterns) > 1:
        #     prev_pattern = patterns[-2:][0]
        #     intersection = set(out_pattern1.value) & set(prev_pattern.value)
        #     # print(len(intersection))

    precision = (num_trials - false_positives) / num_trials
    recall = (num_trials - false_negatives) / num_trials

    f1 = 2 * precision * recall / (recall + precision)
    print(f'f1: {f1}, precision: {precision}, recall: {recall}')

    print(len(area.highway_connections))

    # intersection = set(out_pattern1.value) & set(out_pattern2.value)
    # print(len(intersection))


if __name__ == '__main__':
    main()
