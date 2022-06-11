from typing import Callable


def blackjack_hand_greater_than(hand_1: list, hand_2: list):
    """
    Return True if hand_1 beats hand_2, and False otherwise.

    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21

    Hands are represented as a list of cards. Each card is represented by a string.

    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.

    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.

    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """

    def cal_total(hand: list):
        def fil(with_a: bool) -> Callable[[str], bool]:
            return lambda x: with_a if x == "A" else not with_a

        num_of_a = len([x for x in filter(fil(True), hand)])
        hand_1_no_a = filter(fil(False), hand)

        total_score = 0

        for card in hand_1_no_a:
            total_score += (10 if card in ["J", "Q", "K"] else int(card))

        for i in range(num_of_a):
            total_score += (1 if total_score + 11 > 21 else 11)

        return total_score

    a_total = cal_total(hand_1)
    b_total = cal_total(hand_2)

    return a_total <= 21 and (a_total > b_total or b_total > 21)


# Check your answer
# q3.check()

print(blackjack_hand_greater_than(['K', 'A', '1', '9'], ['3', '4', 'A', '2']))
